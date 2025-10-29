"""
    NAS_DB
    ======

    This module contains the NAS_DB class for accessing the NAS storage.
"""

import sys
import logging
from pathlib import Path

# Add ifi package to Python path for IDE compatibility
current_dir = Path(__file__).resolve()
ifi_parents = [p for p in ([current_dir] if current_dir.is_dir() and current_dir.name=='ifi' else []) 
                + list(current_dir.parents) if p.name == 'ifi']
IFI_ROOT = ifi_parents[-1] if ifi_parents else None

try:
    sys.path.insert(0, str(IFI_ROOT))
except Exception as e:
    logging.error(f"Could not find ifi package root: {e}")
    pass

import configparser
import os
import time
import numpy as np
import pandas as pd
import paramiko
import glob
from typing import Dict, List, Union, Set
import h5py
import re
import tempfile
import threading


from ifi.utils.common import LogManager, ensure_str_path

# Define the set of file extensions that the system is designed to process.
# This prevents attempts to read unsupported files like images (.tif) or documents.
ALLOWED_EXTENSIONS = ['.csv', '.dat', '.mat', '.isf', '.wfm']

# This script finds files and returns a newline-separated list.
# It now accepts multiple patterns separated by spaces.
REMOTE_LIST_SCRIPT = r"""
import sys
import os
import glob
def find_files(base_path_str, patterns_str):
    base_paths = base_path_str.split(';')
    patterns = patterns_str.split(' ')
    all_file_paths = set()
    for base_path in base_paths:
        for pattern in patterns:
            search_pattern = os.path.join(base_path, '**', pattern)
            file_paths = glob.glob(search_pattern, recursive=True)
            all_file_paths.update(file_paths)
    
    sorted_paths = sorted(list(all_file_paths))
    for path in sorted_paths:
        print(path)

if __name__ == "__main__":
    base_path_str = sys.argv[1]
    patterns_str = sys.argv[2]
    find_files(base_path_str, patterns_str)
"""

# This script will be written to the remote machine to read the top N lines of a file.
REMOTE_HEAD_SCRIPT = r"""
import sys
import os

def get_top_lines(file_path, lines_to_read):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i in range(lines_to_read):
                line = f.readline()
                if not line: break
                print(line, end='')
    except Exception as e:
        print(f"Error reading file {file_path}: {e}", file=sys.stderr)

if __name__ == "__main__":
    file_path = sys.argv[1]
    lines_to_read = int(sys.argv[2])
    get_top_lines(file_path, lines_to_read)
"""

LogManager()

class NAS_DB:
    """
    Handles data access from a NAS storage, either directly via a mounted
    network drive or remotely via an SSH connection.
    """
    def __init__(self, config_path='ifi/config.ini'):
        if not Path(config_path).exists():
            raise FileNotFoundError(f"  Config file not found: '{config_path}'  ".center(80, '='))

        config = configparser.ConfigParser()
        config.read(config_path)

        # NAS Config
        nas_cfg = config['NAS']
        self.nas_path = nas_cfg.get('path')
        self.nas_mount = nas_cfg.get('mount_point')
        self.nas_user = nas_cfg.get('user')
        self.nas_password = nas_cfg.get('password')
        
        # Read default data folders, split by comma, and strip whitespace
        folders_str = nas_cfg.get('data_folders', '')
        self.default_data_folders = [f.strip() for f in folders_str.split(',') if f.strip()]

        # SSH Config
        ssh_cfg = config['SSH_NAS']
        self.ssh_host = ssh_cfg.get('ssh_host')
        self.ssh_port = ssh_cfg.getint('ssh_port')
        self.ssh_user = ssh_cfg.get('ssh_user')
        self.ssh_pkey = Path(ssh_cfg.get('ssh_pkey_path')).expanduser()
        self.remote_temp_dir = ssh_cfg.get('remote_temp_dir', None) # Read optional from config

        # Connection settings
        conn_cfg = config['CONNECTION_SETTINGS']
        self.ssh_max_retries = conn_cfg.getint('ssh_max_retries', 3)
        self.ssh_connect_timeout = conn_cfg.getfloat('ssh_connect_timeout', 10.0)

        # Local Cache Config
        if config.has_section('LOCAL_CACHE'):
            cache_cfg = config['LOCAL_CACHE']
            self.dumping_folder = cache_cfg.get('dumping_folder', './cache')
            self.max_load_size_gb = cache_cfg.getfloat('max_load_size_gb', 1.0)
        else:
            self.dumping_folder = './cache'
            self.max_load_size_gb = 1.0

        self.ssh_client = None
        self.sftp_client = None
        self.access_mode = None  # 'local' or 'remote'
        self._file_cache = {} # Cache for file paths for find_files method

        self._is_connected = False
        self.ssh_lock = threading.Lock()
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)

    def _ensure_remote_dir_exists(self, remote_path: str):
        """
        Ensures a directory exists on the remote server using SFTP.
        """
        if self.access_mode != 'remote' or not self.sftp_client:
            return

        # If remote_temp_dir is not set in config, create it in the user's home dir.
        if not remote_path:
            try:
                home_dir = self.sftp_client.normalize('.')
                remote_path = os.path.join(home_dir, '.ifi_temp').replace('\\', '/')
                self.remote_temp_dir = remote_path # Store for later use
                self.logger.info(f"  Remote temp directory not configured, using dynamic path: {remote_path}  ".center(80, '='))
            except Exception as e:
                self.logger.error(f"  Could not determine remote home directory: {e}  ".center(80, '='))
                raise

        try:
            # Check if the path exists and is a directory
            self.sftp_client.stat(remote_path)
        except FileNotFoundError:
            # If it doesn't exist, create it
            self.logger.info(f"Remote directory '{remote_path}' not found. Creating it.")
            self.sftp_client.mkdir(remote_path)
        except Exception as e:
            self.logger.error(f"Could not check or create remote directory '{remote_path}': {e}")
            # Depending on the desired robustness, you might want to raise this exception
            raise

    def connect(self):
        """
        Establishes connection. Checks for local access first, then SSH. Thread-safe.
        """
        if self._is_connected:
            return True

        with self.ssh_lock:
            # Double-check inside the lock to prevent a race condition
            if self._is_connected:
                return True

            if Path(self.nas_mount).is_dir():
                self.logger.info(f"Local NAS mount found at '{self.nas_mount}'. Using direct access.")
                self.access_mode = 'local'
                self._is_connected = True
                return True
            
            self.logger.info("Local NAS mount not found. Attempting SSH connection.")
            self.access_mode = 'remote'
            
            for attempt in range(self.ssh_max_retries):
                try:
                    self.logger.info(f"SSH connection attempt {attempt + 1}/{self.ssh_max_retries}...")
                    self.ssh_client = paramiko.SSHClient()
                    self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    self.ssh_client.connect(
                        hostname=self.ssh_host,
                        port=self.ssh_port,
                        username=self.ssh_user,
                        key_filename=str(self.ssh_pkey),
                        timeout=self.ssh_connect_timeout
                    )
                    self.sftp_client = self.ssh_client.open_sftp()
                    self.logger.info(f"SSH connection to {self.ssh_host} successful.")
                    
                    self._authenticate_nas_remote() 
                    self._is_connected = True
                    return True

                except Exception as e:
                    self.logger.error(f"SSH connection attempt {attempt + 1} failed: {e}")
                    self.disconnect() # disconnect sets _is_connected to False
                    if attempt < self.ssh_max_retries - 1:
                        self.logger.info("Retrying in 3 seconds...")
                        time.sleep(3)
            
            self._is_connected = False
            self.logger.error("All SSH connection attempts failed.")
            return False

    def _authenticate_nas_remote(self):
        """
        Runs 'net use' on the remote machine to authenticate with the NAS.
        """
        auth_cmd = f'net use "{str(self.nas_path)}" /user:{self.nas_user} {self.nas_password} /persistent:no'
        self.logger.info("Authenticating to NAS on remote machine.")
        stdin, stdout, stderr = self.ssh_client.exec_command(auth_cmd)
        
        exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            err = stderr.read().decode('utf-8', errors='ignore').strip()
            # "The command completed successfully." is not an error.
            if "The command completed successfully" not in err:
                 self.logger.warning(f"NAS authentication may have failed with exit code {exit_status}: {err}")
            else:
                self.logger.info("NAS authentication successful.")
        else:
            self.logger.info("NAS authentication successful.")


    def find_files(self, query: Union[int, str, List[Union[int, str]]], data_folders: List[str] = None, add_path: bool = False, force_remote: bool = False) -> List[str]:
        """
        Finds all files matching a query across multiple folders.

        The query can be:
        - A single shot number (int): 45821
        - A list of shot numbers (List[int]): [45821, 45822]
        ┌ A single file pattern (str): "45821_*.csv"
        ├ In case of numbers in the file name(str): "45821_[0-9]+.csv"
        └ In case of characters in the file name (str): "45821_[a-zA-Z]+.csv"
        - A list of file patterns (List[str]): ["45821_*.csv", "45822_*.dat"]
        - A single full file path (str): "/path/to/your/file.csv"
        - A list of full file paths (List[str]): ["/path/to/file1.csv", "/path/to/file2.csv"]

        If `data_folders` is not provided, it uses the default folders from config.ini.
        The method caches results based on the query and data_folders.
        """
        if not self._is_connected:
            if not self.connect():
                raise ConnectionError("Failed to establish connection to NAS.")

        if data_folders is not None:
            if isinstance(data_folders, str):
                data_folders = [data_folders]
            elif isinstance(data_folders, list):
                for i, folder in enumerate(data_folders):
                    if not isinstance(folder, str):
                        self.logger.warning(f"Invalid data folder type: {type(folder)}. Converting to string. {folder}")
                        data_folders[i] = str(folder)
            else:
                self.logger.warning(f"Invalid data folder type: {type(data_folders)}. {data_folders}")
                data_folders = [data_folders]
        else:
            data_folders = self.default_data_folders

        if add_path:
            data_folders = list(set(self.default_data_folders + data_folders))

        cache_key = (str(query), tuple(sorted(data_folders)))
        if cache_key in self._file_cache:
            self.logger.info(f"Found file list for {cache_key} in cache.")
            return self._file_cache[cache_key]

        self.logger.info(f"Searching for files for query: {query} in folders: {data_folders}.")

        base_path = str(self.nas_mount) if self.access_mode == 'local' else str(self.nas_path)
        
        # --- Handle full paths directly ---
        if isinstance(query, str) and (os.path.sep in query or '/' in query):
             # A single full path is passed
             return [query] if Path(query).exists() else []
        if isinstance(query, list) and all(isinstance(q, str) and (os.path.sep in q or '/' in q) for q in query):
            # A list of full paths is passed
            return [q for q in query if Path(q).exists()]

        # --- Build search patterns for shot numbers and wildcards ---
        query_items = query if isinstance(query, list) else [query]
        search_patterns = []
        for item in query_items:
            # Add wildcards for all extensions if it's just a number or simple string
            if isinstance(item, int) or (isinstance(item, str) and '*' not in item and '.' not in item):
                 search_patterns.append(f'{item}*.*')
            else: # It's already a pattern like "45821_*.csv"
                 search_patterns.append(str(item))
        
        all_files: Set[str] = set()
        if self.access_mode == 'local':
            for folder in data_folders:
                for pattern in search_patterns:
                    search_path = Path(base_path) / folder / '**' / pattern
                    found = glob.glob(str(search_path), recursive=True)
                    all_files.update(found)
        else: # remote access
            all_found_paths = self._find_files_remote(data_folders, search_patterns)
            all_files.update(all_found_paths)

        sorted_files = sorted(list(all_files))

        # --- Filter by allowed extensions ---
        filtered_files = [
            f for f in sorted_files
            if Path(f).suffix.lower() in ALLOWED_EXTENSIONS
        ]
        if len(filtered_files) < len(sorted_files):
            self.logger.info(
                f"Filtered file list from {len(sorted_files)} to {len(filtered_files)} "
                f"based on allowed extensions: {ALLOWED_EXTENSIONS}"
            )

        # --- Normalize paths to use forward slashes for consistency ---
        normalized_files = [f.replace('\\', '/') for f in filtered_files]

        if normalized_files:
            self.logger.info(f"Found {len(normalized_files)} files. Caching result.")
            self._file_cache[cache_key] = normalized_files
        else:
            self.logger.warning(f"No files with allowed extensions found for query '{query}' in {data_folders}. Caching empty list.")
            self._file_cache[cache_key] = []
            
        return normalized_files

    def _find_files_remote(self, data_folders: List[str], patterns: List[str]) -> List[str]:
        """ Executes a single remote script to find files across multiple folders and patterns. """
        patterns_str = ' '.join(patterns)

        # Combine all search folders into a single semicolon-separated string
        search_paths = [os.path.join(self.nas_path, folder).replace('\\', '/') for folder in data_folders]
        search_paths_str = ';'.join(search_paths)

        # Ensure the remote temp directory exists before trying to write to it
        self._ensure_remote_dir_exists(self.remote_temp_dir)

        remote_script_path = os.path.join(self.remote_temp_dir, f'list_{int(time.time())}.py').replace('\\', '/')
        
        try:
            with self.sftp_client.open(remote_script_path, 'w') as f:
                f.write(REMOTE_LIST_SCRIPT)
        except Exception as e:
            self.logger.error(f"Failed to write remote list script: {e}")
            return []

        cmd = f'python "{remote_script_path}" "{search_paths_str}" "{patterns_str}"'
        stdin, stdout, stderr = self.ssh_client.exec_command(cmd)

        files = stdout.read().decode('utf-8').strip().splitlines()
        
        err_output = stderr.read().decode('utf-8', errors='ignore').strip()
        if err_output:
            self.logger.error(f"Remote list script error: {err_output}")
    
        return files

    def _get_files_total_size(self, file_list: List[str]) -> int:
        """Calculates the total size of a list of files in bytes."""
        total_size = 0
        if not file_list:
            return 0
        
        self.logger.info(f"Calculating total size of {len(file_list)} files...")
        
        try:
            if self.access_mode == 'local':
                for file_path in file_list:
                    if Path(file_path).exists():
                        total_size += Path(file_path).stat().st_size
            else: # remote
                for file_path in file_list:
                    try:
                        total_size += self.sftp_client.stat(file_path).st_size
                    except FileNotFoundError:
                        self.logger.warning(f"Remote file not found during size calculation: {file_path}")
                        continue
            
            return total_size
        except Exception as e:
            self.logger.error(f"Error calculating file sizes: {e}")
            return -1 # Return -1 to indicate an error

    def get_shot_data(self, query: Union[int, str, List[Union[int, str]]], data_folders: Union[list, str] = None, add_path: bool = False, force_remote: bool = False, **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Retrieves data for a given shot number, pattern, or list of files.
        Caches each file to a dedicated HDF5 file based on its shot number.

        Args:
            query: A shot number, list of shot numbers, a search pattern (e.g., "45*"), 
                   or a list of specific file paths.
            data_folders: The specific subfolder(s) to search. If None, uses defaults from config.
            force_remote: If True, bypasses the local cache and fetches from the NAS.
            **kwargs: Additional arguments passed to the data parsing functions (e.g., skiprows).
        
        Returns:
            A dictionary mapping each successfully read filename to its DataFrame.
        """
        if not self._is_connected:
            if not self.connect():
                raise ConnectionError("Failed to establish connection to NAS.")
        
        # --- Find all target files on the NAS ---
        target_files = self.find_files(query, data_folders, add_path, force_remote, **kwargs)
        if not target_files:
            self.logger.warning(f"No files found on NAS for query: {query}")
            return {}

        data_dict: Dict[str, pd.DataFrame] = {}
        files_to_fetch = []

        if force_remote:
            files_to_fetch = target_files
        else:
            # --- Check cache for each file individually ---
            for file_path in target_files:
                basename = Path(file_path).name
                # Try to extract shot number from filename (e.g., "45821_056.csv" -> "45821")
                match = re.match(r'(\d+)', basename)
                shot_num_for_cache = int(match.group(1)) if match else None

                if shot_num_for_cache is None:
                    self.logger.warning(f"Could not determine shot number for '{basename}'. Will not use cache.")
                    files_to_fetch.append(file_path)
                    continue
                
                cache_dir = Path(self.dumping_folder) / str(shot_num_for_cache)
                cache_file = cache_dir / f'{shot_num_for_cache}.h5'
                
                # --- Start Diagnostic Logging ---
                self.logger.info(f"[Cache Check] For file: '{basename}'")
                self.logger.info(f"[Cache Check] Checking for cache file at: '{cache_file.absolute()}'")
                
                if not cache_file.exists():
                    self.logger.warning("[Cache Check] -> Cache file NOT FOUND.")
                    files_to_fetch.append(file_path)
                    continue
                
                self.logger.info("[Cache Check] -> Cache file FOUND.")
                # --- End Diagnostic Logging ---
                
                # Sanitize the basename to be a valid HDF5 key, matching the writing logic.
                key = re.sub(r'[^a-zA-Z0-9_]', '_', basename)
                if key and not key[0].isalpha() and not key.startswith('_'):
                    key = '_' + key
                
                # --- Start Diagnostic Logging ---
                self.logger.info(f"[Cache Check] Looking for key: '{key}'")
                # --- End Diagnostic Logging ---

                try:
                    with h5py.File(cache_file, 'r') as f:
                        if key in f:
                            self.logger.info("[Cache Check] -> Key FOUND in cache. Loading from HDF5.")
                            self.logger.info(f"Found '{basename}' in cache: {cache_file}")
                            df = pd.read_hdf(cache_file, key)
                            
                            # Restore metadata if available
                            metadata_key = f"{key}_metadata"
                            if metadata_key in f:
                                self.logger.info(f"Restoring metadata for '{basename}' from key '{metadata_key}'")
                                metadata_series = pd.read_hdf(cache_file, metadata_key)
                                metadata_dict = metadata_series.to_dict()
                                
                                # JSON 문자열을 다시 딕셔너리로 변환
                                import json
                                for k, v in metadata_dict.items():
                                    if k == 'metadata' and isinstance(v, str):
                                        try:
                                            df.attrs[k] = json.loads(v)
                                        except:
                                            df.attrs[k] = v
                                    else:
                                        df.attrs[k] = v
                                
                                self.logger.info(f"Restored metadata: {df.attrs}")
                            
                            data_dict[file_path] = df
                        else:
                            self.logger.warning("[Cache Check] -> Key NOT FOUND in cache file. Re-fetching.")
                            files_to_fetch.append(file_path)
                except Exception as e:
                    self.logger.error(f"Error reading cache file '{cache_file}': {e}. Will refetch.")
                    files_to_fetch.append(file_path)

        if not files_to_fetch:
            self.logger.info("All required files were found in the cache.")
            return data_dict

        # --- Memory Check ---
        if self.max_load_size_gb > 0:
            total_size_bytes = self._get_files_total_size(files_to_fetch)
            total_size_gb = total_size_bytes / (1024**3)
            
            if total_size_bytes < 0:
                 self.logger.warning("Could not determine total size of files to fetch. Proceeding with caution.")
            elif total_size_gb > self.max_load_size_gb:
                raise MemoryError(
                    f"Total size of files to fetch ({total_size_gb:.2f} GB) exceeds the configured limit "
                    f"of {self.max_load_size_gb:.2f} GB. To override, increase 'max_load_size_gb' in "
                    f"config.ini or use a more specific query."
                )
            else:
                self.logger.info(f"Total size of files to fetch: {total_size_gb:.2f} GB. (Limit: {self.max_load_size_gb:.2f} GB)")

        # --- Fetching and Caching Logic ---
        self.logger.info(f"Fetching {len(files_to_fetch)} files from NAS...")
        for file_path in files_to_fetch:
            df = self._read_shot_file(file_path, **kwargs)
            if df is not None:
                data_dict[file_path] = df
                
                # --- Cache the newly fetched file ---
                basename = Path(file_path).name
                match = re.match(r'(\d+)', basename)
                shot_num_for_cache = int(match.group(1)) if match else None

                if shot_num_for_cache is not None:
                    cache_dir = Path(self.dumping_folder) / str(shot_num_for_cache)
                    cache_file = cache_dir / f'{shot_num_for_cache}.h5'
                    
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    # Sanitize the basename to be a valid HDF5 key
                    key = re.sub(r'[^a-zA-Z0-9_]', '_', basename)
                    if key and not key[0].isalpha() and not key.startswith('_'):
                        key = '_' + key
                    self.logger.info(f"Caching '{basename}' to '{cache_file}' with key '{key}'")
                    # Use keyword arguments for to_hdf for future compatibility with pandas 3.0
                    df.to_hdf(path_or_buf=cache_file, key=key, mode='a', format='table', complevel=5, complib='zlib')
                    
                    # Save metadata separately to preserve attrs
                    if hasattr(df, 'attrs') and df.attrs:
                        metadata_key = f"{key}_metadata"
                        # 메타데이터를 문자열로 변환하여 저장
                        metadata_dict = {}
                        for k, v in df.attrs.items():
                            if isinstance(v, dict):
                                # 중첩된 딕셔너리는 JSON 문자열로 변환
                                import json
                                metadata_dict[k] = json.dumps(v)
                            else:
                                metadata_dict[k] = str(v)
                        
                        metadata_series = pd.Series(metadata_dict)
                        metadata_series.to_hdf(path_or_buf=cache_file, key=metadata_key, mode='a', format='table', complevel=5, complib='zlib')
                        self.logger.info(f"Cached metadata for '{basename}' with key '{metadata_key}'")
                else:
                    self.logger.warning(f"Could not determine shot number for '{basename}'. Skipping cache.")

        # Sort the final dictionary by key (filename) for consistent order
        return dict(sorted(data_dict.items()))

    def _read_shot_file(self, file_path: str, **kwargs) -> pd.DataFrame | None:
        """
        Master parser that dispatches to the correct reader based on file extension.
        This method is a thread-safe wrapper for remote operations.
        """
        if self.access_mode == 'remote':
            with self.ssh_lock:
                # Ensures that each dask task gets exclusive access to the SSH client.
                return self._dispatch_read(file_path, **kwargs)
        else:
            # No lock needed for local access
            return self._dispatch_read(file_path, **kwargs)

    def _dispatch_read(self, file_path: str, **kwargs) -> pd.DataFrame | None:
        """
        Internal dispatcher for reading files. Assumes any necessary locks are held.
        """
        ext = Path(file_path).suffix.lower()

        if ext == '.csv':
            return self._read_scope_csv(file_path, **kwargs)
        elif ext == '.dat':
            return self._read_fpga_dat(file_path, **kwargs)
        elif ext == '.mat':
            return self._read_matlab_mat(file_path, **kwargs)
        else:
            self.logger.warning(f"Unsupported file extension '{ext}' for file: {file_path}")
            return None

    def _read_scope_csv(self, file_path: str, **kwargs) -> pd.DataFrame | None:
        """
        Reads a CSV file by first identifying its type from the header.
        Thread-safe for network drives with file-level locking.
        """
        self.logger.info(f"Reading CSV file: {file_path}")
        
        # Get header without any locks for full parallel processing
        header_text = None
        if self.access_mode == 'local':
            header_text = self._get_data_top_local(file_path, lines=40)
        else: # remote
            header_text = self._get_data_top_remote(file_path, lines=40)

        if not header_text:
            self.logger.error(f"Could not read header of {file_path}")
            return None

        header_content = header_text.splitlines()
        csv_type = self._identify_csv_type(header_content)
        self.logger.info(f"Identified CSV type for {Path(file_path).name} as: {csv_type}")

        # All files run in parallel - no locks
        try:
            df = self._parse_csv_with_type(csv_type, file_path, header_content, **kwargs)
            
            if df is not None:
                df.attrs['source_file_type'] = 'csv'
                df.attrs['source_file_format'] = csv_type
                self.logger.info(f"Successfully parsed file: {file_path}")
                return df  # Success, return immediately
        except Exception as read_error:
            self.logger.error(f"Failed to parse file '{file_path}'. Error: {read_error}")
            return None

    def _parse_csv_with_type(self, csv_type: str, file_path: str, header_content: list, **kwargs) -> pd.DataFrame:
        """
        Parse CSV file based on its identified type.
        """
        if csv_type == 'MDO3000pc':
            return self._parse_mdo3000pc(file_path, header_content, **kwargs)
        elif csv_type == 'MSO58':
            return self._parse_mso58(file_path, header_content, **kwargs)
        elif csv_type in ['MDO3000orig', 'MDO3000fetch', 'ETC']:
            return self._parse_standard_csv(file_path, header_content, csv_type, **kwargs)
        else:
            self.logger.error(f"Unknown CSV type '{csv_type}' for {file_path}")
            return None

    def _identify_csv_type(self, header_content: list[str]) -> str:
        """
        Identifies the CSV format based on the first 40 lines of content.
        """
        # Placeholder for the complex identification logic
        # For now, returning ETC as a default
        try:
            lines = [line.strip().split(',') for line in header_content]
            if not lines: return 'UNKNOWN'

            # MDO3000 / MSO58 Check
            if 'model' in lines[0][0].lower():
                if len(lines[0]) > 1:
                    model_name = lines[0][1]
                    if 'MDO3' in model_name:
                        if len(lines) > 1 and 'firmware version' in lines[1][0].lower():
                            return 'MDO3000orig'
                        else:
                            return 'MDO3000fetch'
                    elif 'MSO5' in model_name:
                        return 'MSO58'
            
            # MDO3000pc Check
            if len(lines) > 16 and len(lines[16]) > 1 and 'MDO3' in lines[16][1]:
                return 'MDO3000pc'

        except IndexError:
            # File is likely not in a recognized format, treat as ETC
            pass
            
        return 'ETC'

    def _parse_mdo3000pc(self, file_path: str, header_content: list[str], **kwargs) -> pd.DataFrame | None:
        self.logger.info(f"Parsing as MDO3000pc: {file_path}")
        
        metadata = {}
        source_line_content = []
        source_line_index = -1

        # 1. Find metadata and the 'Source' line
        for i, line in enumerate(header_content):
            parts = line.strip().split(',')
            if len(parts) > 1:
                key = parts[0].strip()
                val = parts[1].strip()
                if 'Record Length' in key:
                    metadata['record_length'] = int(val)
                elif 'Sample Interval' in key:
                    metadata['time_resolution'] = float(val)
            
            if 'Source' in line:
                source_line_content = parts
                source_line_index = i

        if source_line_index == -1:
            self.logger.error(f"Could not find 'Source' line in header for MDO3000pc file: {file_path}")
            return None

        self.logger.info(f"MDO3000pc 'Source' line content: {source_line_content}") # Debugging line

        # 2. Determine column names and data column indices from 'Source' line
        ch_names = []
        data_col_indices = []
        for i, part in enumerate(source_line_content):
            if 'Source' in part and i + 1 < len(source_line_content):
                ch_names.append(source_line_content[i+1].strip())
            
            # Make parsing more robust - try converting to float
            try:
                float(part.strip())
                data_col_indices.append(i)
            except ValueError:
                continue

        # This is interpreted as taking the 1st, 2nd, 4th... (i.e., 1st and even-indexed) numeric columns
        final_data_indices = [data_col_indices[0]]
        final_data_indices.extend([data_col_indices[i] for i in range(len(data_col_indices)) if i % 2 == 1]) 
        
        if not final_data_indices:
            self.logger.error(f"Could not extract data column indices for MDO3000pc file: {file_path}")
            return None

        # 3. Read the file with pandas, using the determined indices
        read_target = file_path
        temp_file = None
        try:
            # The actual data starts after the header block (assumed to be 17 lines)
            if self.access_mode == 'remote':
                self.logger.info(f"Downloading remote file to temp location: {file_path}")
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
                self.sftp_client.get(file_path, temp_file.name)
                read_target = temp_file.name
            else:
                # Use pathlib.Path for safe local file handling
                read_target = Path(read_target)
            
            # Prefer C engine; on failure, fallback to python engine (without low_memory)
            try:
                p = ensure_str_path(read_target)
                # Use file handle to avoid Windows path quirks under threaded IO
                with open(p, 'rb') as fh:
                    df = pd.read_csv(
                        fh,
                        skiprows=17,
                        header=None,
                        usecols=final_data_indices,
                        low_memory=False,
                        on_bad_lines='warn',
                        encoding_errors='ignore',
                        memory_map=False
                    )
            except Exception as e_c:
                self.logger.warning(f"C engine parse failed for MDO3000pc: {e_c}. Retrying with python engine.")
                p = ensure_str_path(read_target)
                with open(p, 'rb') as fh:
                    df = pd.read_csv(
                        fh,
                        skiprows=17,
                        header=None,
                        usecols=final_data_indices,
                        on_bad_lines='warn',
                        encoding_errors='ignore',
                        engine='python'
                    )
            
            # Standardize column names: TIME for first column, CH0, CH1, CH2... for the rest
            standardized_cols = ['TIME'] + [f'CH{i}' for i in range(len(df.columns)-1)]
            df.columns = standardized_cols
            
            df.attrs['metadata'] = metadata
            return df

        except Exception as e:
            self.logger.error(f"Pandas failed to parse MDO3000pc file {file_path}. Error: {e}")
            return None
        finally:
            if temp_file:
                temp_file.close()
                os.unlink(temp_file.name)


    def _parse_mso58(self, file_path: str, header_content: list[str], **kwargs) -> pd.DataFrame | None:
        """
        Parses data from a Tektronix MSO58 series oscilloscope CSV file.
        This format has a fixed header length and no data header row.
        """
        self.logger.info(f"Parsing as MSO58: {file_path}")
        
        metadata = {}
        header_row_index = -1

        # 1. Dynamically find the header row, similar to _parse_standard_csv
        for i, line in enumerate(header_content):
            # Parse metadata from any line
            try:
                parts = line.strip().split(',')
                if len(parts) > 1:
                    key = parts[0].strip()
                    val = parts[1].strip()
                    if 'Record Length' in key:
                        metadata['record_length'] = int(val)
                    elif 'Sample Interval' in key:
                        metadata['time_resolution'] = float(val)
            except (ValueError, IndexError):
                pass # Ignore lines that can't be parsed for metadata

            # Find the actual header row
            if 'TIME' in line.upper() and 'CH' in line.upper():
                header_row_index = i
                self.logger.info(f"Dynamically detected MSO58 header at line: {header_row_index}")
                break
        
        if header_row_index == -1:
            self.logger.error(f"Could not dynamically determine the header row for MSO58 file: {file_path}")
            return None

        read_target = file_path
        temp_file = None
        try:
            if self.access_mode == 'remote':
                self.logger.info(f"Downloading remote file to temp location: {file_path}")
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
                self.sftp_client.get(file_path, temp_file.name)
                read_target = temp_file.name
            else:
                # Use pathlib.Path for safe local file handling
                read_target = Path(read_target)

            df = pd.read_csv(
                read_target,
                skiprows=header_row_index,
                header=0, # Use the found line as the header
                encoding_errors='ignore',
                engine='python',
                skipfooter=0
            )

            # Standardize column names: TIME for first column, CH0, CH1, CH2... for the rest
            standardized_cols = ['TIME'] + [f'CH{i}' for i in range(len(df.columns)-1)]
            df.columns = standardized_cols

            # Verification log
            actual_len = len(df)
            expected_len = metadata.get('record_length')
            self.logger.info(f"File: {Path(file_path).name}, Record Length from header: {expected_len}, Actual rows read: {actual_len}")
            if expected_len is not None and expected_len != actual_len:
                self.logger.warning(f"RECORD LENGTH MISMATCH for {Path(file_path).name}! Header: {expected_len}, Actual: {actual_len}")

            df.attrs['metadata'] = metadata
            return df
        except Exception as e:
            self.logger.error(f"Pandas failed to parse MSO58 file {file_path}. Error: {e}")
            return None
        finally:
            if temp_file:
                temp_file.close()
                os.unlink(temp_file.name)

    def _parse_standard_csv(self, file_path: str, header_content: list[str], csv_type: str, **kwargs) -> pd.DataFrame | None:
        self.logger.info(f"Parsing as {csv_type}: {file_path}")
        
        metadata = {}
        header_row_index = -1
        
        # 1. Find metadata and header row from the initial content
        for i, line in enumerate(header_content):
            try:
                parts = line.strip().split(',')
                if len(parts) > 1:
                    key = parts[0].strip()
                    val = parts[1].strip()
                    if 'Record Length' in key:
                        metadata['record_length'] = int(val)
                    elif 'Sample Interval' in key:
                        metadata['time_resolution'] = float(val)
                
                # Find the actual header row
                if 'TIME' in line.upper() and 'CH' in line.upper():
                    header_row_index = i
            except (ValueError, IndexError):
                # Ignore lines that don't fit the key-value or header pattern
                continue

        if header_row_index == -1:
            self.logger.error(f"Could not find a valid header row ('TIME', 'CH...') in {file_path}")
            return None
            
        self.logger.info(f"Found header at line {header_row_index + 1}. Metadata: {metadata}")

        # 2. Read the actual data using pandas, skipping to the data section
        read_target = file_path
        temp_file = None
        try:
            if self.access_mode == 'remote':
                self.logger.info(f"Downloading remote file to temp location: {file_path}")
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
                self.sftp_client.get(file_path, temp_file.name)
                read_target = temp_file.name
            else:
                # Use pathlib.Path for safe local file handling
                read_target = Path(read_target)
            
            # Treat the detected header line as the header row (align with MSO58 handling)
            # Try C engine first; fallback to python engine if needed
            try:
                p = ensure_str_path(read_target)
                with open(p, 'rb') as fh:
                    df = pd.read_csv(
                        fh,
                        skiprows=header_row_index,
                        header=0,
                        encoding='utf-8',
                        on_bad_lines='warn',
                        low_memory=False,
                        encoding_errors='ignore',
                        memory_map=False
                    )
            except Exception as e_c:
                self.logger.warning(f"C engine parse failed for standard CSV: {e_c}. Retrying with python engine.")
                p = ensure_str_path(read_target)
                with open(p, 'rb') as fh:
                    df = pd.read_csv(
                        fh,
                        skiprows=header_row_index,
                        header=0,
                        encoding='utf-8',
                        on_bad_lines='warn',
                        encoding_errors='ignore',
                        engine='python'
                    )
 
            # Standardize column names: TIME for first column, CH0, CH1, CH2... for the rest
            standardized_cols = ['TIME'] + [f'CH{i}' for i in range(len(df.columns)-1)]
            df.columns = standardized_cols
            
            # Add metadata to the DataFrame's attributes for later use
            df.attrs['metadata'] = metadata
            
            return df
        except Exception as e:
            self.logger.error(f"Pandas failed to parse {file_path} with detected header. Error: {e}")
            return None
        finally:
            if temp_file:
                temp_file.close()
                os.unlink(temp_file.name)

    def _read_fpga_dat(self, file_path: str, **kwargs) -> pd.DataFrame | None:
        self.logger.info(f"Parsing as FPGA .dat: {file_path}")
        read_target = file_path
        temp_file = None
        try:
            if self.access_mode == 'remote':
                self.logger.info(f"Downloading remote .dat file to temp location: {file_path}")
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.dat')
                self.sftp_client.get(file_path, temp_file.name)
                read_target = temp_file.name
            else:
                # Use pathlib.Path for safe local file handling
                read_target = Path(read_target)

            # Read data first without column names
            p = ensure_str_path(read_target)
            with open(p, 'rb') as fh:
                df = pd.read_csv(fh, sep=r'\s+', header=None)
            
            # Standardize column names: TIME for first column, CH0, CH1, CH2... for the rest
            standardized_cols = ['TIME'] + [f'CH{i}' for i in range(len(df.columns)-1)]
            df.columns = standardized_cols
            df.attrs['source_file_type'] = 'dat'
            df.attrs['source_file_format'] = 'FPGA'
            return df
        except Exception as e:
            self.logger.error(f"Failed to parse FPGA file {file_path}: {e}")
            return None
        finally:
            if temp_file:
                temp_file.close()
                os.unlink(temp_file.name)

    def _read_matlab_mat(self, file_path: str, **kwargs) -> pd.DataFrame | None:
        self.logger.info(f"Parsing as MATLAB .mat: {file_path}")
        try:
            local_mat_path = file_path
            if self.access_mode == 'remote':
                # .mat are binary, need to be downloaded whole first.
                self.logger.warning("Remote .mat files will be downloaded to a temporary local file.")
                local_mat_path = Path(self.dumping_folder) / Path(file_path).name
                self.sftp_client.get(file_path, local_mat_path)
            
            from scipy.io import loadmat
            mat = loadmat(local_mat_path)

            if self.access_mode == 'remote':
                os.remove(local_mat_path) # Clean up temp file

            # This assumes a simple structure, might need adjustment based on actual .mat files
            # Look for a variable that is a numpy array
            data_key = [k for k in mat if isinstance(mat[k], np.ndarray) and not k.startswith('__')][0]
            data = mat[data_key]
            
            cols = ['TIME'] + [f'CH{i+1}' for i in range(data.shape[1] - 1)]
            df = pd.DataFrame(data, columns=cols)
            df.attrs['source_file_type'] = 'mat'
            df.attrs['source_file_format'] = 'MATLAB'
            return df
        except Exception as e:
            self.logger.error(f"Failed to parse MATLAB file {file_path}: {e}")
            return None

    def get_data_top(self, query: Union[int, str, List[Union[int, str]]], data_folders: Union[list, str] = None, lines: int = 50) -> str | None:
        """
        Retrieves the first few lines of the first file found for a shot.
        Useful for inspecting headers.
        """
        if not self._is_connected:
            if not self.connect():
                raise ConnectionError("Failed to establish connection to NAS.")

        if isinstance(data_folders, str):
            data_folders = [data_folders]
            
        file_paths = self.find_files(query, data_folders)
        if not file_paths:
            self.logger.warning(f"No files found for query {query} to get header from.")
            return None
        
        first_file = file_paths[0]
        self.logger.info(f"Getting top {lines} lines from: {first_file}")

        if self.access_mode == 'local':
            return self._get_data_top_local(first_file, lines)
        else: # remote
            return self._get_data_top_remote(first_file, lines)

    def _get_data_top_local(self, file_path: str, lines: int):
        self.logger.info(f"Reading top {lines} lines from local file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                head = [f.readline() for _ in range(lines)]
            return "".join(line for line in head if line)
        except Exception as e:
            self.logger.error(f"Error reading top lines from {file_path}: {e}")
            return None

    def _get_data_top_remote(self, file_path: str, lines: int):
        self.logger.info(f"Reading top {lines} lines from remote file...")

        # 1. Write the head script to a remote temp file
        self._ensure_remote_dir_exists(self.remote_temp_dir)
        remote_script_path = os.path.join(self.remote_temp_dir, f'head_{int(time.time())}.py').replace('\\', '/')
        try:
            with self.sftp_client.open(remote_script_path, 'w') as f:
                f.write(REMOTE_HEAD_SCRIPT)
        except Exception as e:
            self.logger.error(f"Failed to write remote script: {e}")
            return None

        # 2. Execute the script
        cmd = f'python "{remote_script_path}" "{file_path}" "{lines}"'
        
        self.logger.info("Executing remote command...")
        stdin, stdout, stderr = self.ssh_client.exec_command(cmd)

        # 3. Get output and check for errors
        output = stdout.read().decode('utf-8', errors='ignore')
        err_output = stderr.read().decode('utf-8', errors='ignore').strip()
        if err_output:
            self.logger.error(f"Remote script error: {err_output}")
        
        # 4. Cleanup remote script
        try:
            self.sftp_client.remove(remote_script_path)
        except Exception as e:
            self.logger.warning(f"Failed to remove remote script {remote_script_path}: {e}")

        if output:
            return output
        else:
            self.logger.warning("No data returned from remote script for get_data_top.")
            return None

    def disconnect(self):
        if self.sftp_client:
            self.sftp_client.close()
            self.sftp_client = None
        if self.ssh_client:
            self.ssh_client.close()
            self.ssh_client = None
        self._is_connected = False
        self.logger.info("Disconnected.")

    def __enter__(self):
        if not self.connect():
            raise ConnectionError("Failed to establish connection to NAS.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

if __name__ == '__main__':
    # This example demonstrates how to use the NAS_DB class.
    # It requires a valid 'ifi/config.ini' file with [NAS] and [SSH_NAS] sections.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
            
    logging.info("--- Testing NAS_DB ---")
    
    # Example usage:
    shot_to_find = 45821  # Example shot number
    folder_to_search = 'Data/MSO58_2' # Example sub-folder

    try:
        with NAS_DB() as nas:
            # --- New, more efficient workflow ---
            
            # 1. First call: Data is not cached. It will be fetched from the remote source
            #    and then saved to the local cache (e.g., './cache/45821.h5').
            logging.info("\n--- 1. First call to get_shot_data (should fetch and cache) ---")
            data_dict = nas.get_shot_data(shot_to_find, folder_to_search, add_path=True)
            if data_dict:
                logging.info("   -> Data loaded successfully on first call.")
                logging.info(f"   -> Number of dataframes: {len(data_dict)}")
                for key, df in data_dict.items():
                    logging.info(f"   -> DataFrame '{key}' shape: {df.shape}")

            # 2. Second call: The local cache file now exists. This call should be much faster
            #    as it reads directly from the local HDF5 file.
            logging.info("\n--- 2. Second call to get_shot_data (should load from cache) ---")
            df_from_cache = nas.get_shot_data(shot_to_find, folder_to_search, add_path=True)
            if df_from_cache:
                logging.info("   -> Data loaded successfully from cache.")
                logging.info(f"   -> Number of dataframes: {len(df_from_cache)}")
                for key, df in df_from_cache.items():
                    logging.info(f"   -> DataFrame '{key}' shape: {df.shape}")

            # 3. Force remote fetch: Use 'force_remote=True' to bypass the local cache and
            #    re-download the data from the source. This is useful if the data has changed.
            logging.info("\n--- 3. Third call with force_remote=True and add_path=True (should bypass cache) ---")
            df_forced = nas.get_shot_data(shot_to_find, folder_to_search, force_remote=True, add_path=True)
            if df_forced:
                logging.info("   -> Data loaded successfully by forcing remote fetch.")
                logging.info(f"   -> Number of dataframes: {len(df_forced)}")
                for key, df in df_forced.items():
                    logging.info(f"   -> DataFrame '{key}' shape: {df.shape}")

            # 4. Force remote fetch: Use 'force_remote=True' to bypass the local cache and
            #    find the data at the designated folder without appending the sub-folder name.
            logging.info("\n--- 4. Fourth call with force_remote=True but add_path=False (should bypass cache) ---")
            df_forced = nas.get_shot_data(shot_to_find, folder_to_search, force_remote=True, add_path=False)
            if df_forced:
                logging.info("   -> Data loaded successfully by forcing remote fetch.")
                logging.info(f"   -> Number of dataframes: {len(df_forced)}")
                for key, df in df_forced.items():
                    logging.info(f"   -> DataFrame '{key}' shape: {df.shape}")

    except FileNotFoundError as e:
        logging.error(f"Configuration error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True) 