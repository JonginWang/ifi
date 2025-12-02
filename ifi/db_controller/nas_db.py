"""
NAS_DB
======

This module contains the NAS_DB class for accessing the NAS storage.

Classes:
    NAS_DB: The class for accessing the NAS storage.

Variables:
    ALLOWED_EXTENSIONS: The set of file extensions that the system is designed to process.
    REMOTE_LIST_SCRIPT: The script to find files on the remote machine.
    REMOTE_HEAD_SCRIPT: The script to read the top N lines of a file on the remote machine.

Usage Example:
    ```python
    from . import NAS_DB

    # Initialize and use NAS_DB
    with NAS_DB() as nas:
        # Get shot data (will cache automatically)
        # Using if local cache is available, it will use the local cache.
        data_dict = nas.get_shot_data(45821, "ifi/references/data", add_path=True)
        # Using force_remote=True to bypass the local cache and fetch the data
        # from the remote machine.
        data_dict = nas.get_shot_data(45821, "ifi/references/data", add_path=True, force_remote=True)
        # If data_dict is not empty, it will print the shape of the dataframes.
        if data_dict:
            for key, df in data_dict.items():
                print(f"DataFrame '{key}' shape: {df.shape}")
    ```
"""

from pathlib import Path
import logging
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
import uuid
from contextlib import nullcontext
try:
    from ..utils.common import LogManager, ensure_str_path, log_tag
    from .. import get_project_root
except ImportError as e:
    print(f"Failed to import ifi modules: {e}. Ensure project root is in PYTHONPATH.")
    from ifi.utils.common import LogManager, ensure_str_path, log_tag
    from ifi import get_project_root

# Define the set of file extensions that the system is designed to process.
# This prevents attempts to read unsupported files like images (.tif) or documents.
# ALLOWED_EXTENSIONS = [".csv", ".dat", ".mat", ".isf", ".wfm"]
ALLOWED_EXTENSIONS = [".csv", ".dat", ".isf", ".wfm"]

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


## Helper functions
def _generate_unique_script_name(prefix: str = "script") -> str:
    """
    Generate a unique filename for remote scripts to avoid collisions in parallel execution.
    
    Uses UUID to ensure uniqueness across processes and threads.
    
    Args:
        prefix: Prefix for the filename (e.g., "list", "head")
        
    Returns:
        Unique filename string (e.g., "list_a1b2c3d4e5f6.py")
    """
    unique_id = uuid.uuid4().hex[:12]  # 12-character hex string
    return f"{prefix}_{unique_id}.py"


def _is_drive_or_unc_path(path_str: str) -> bool:
    """Check if a string looks like a drive or UNC path."""
    if not isinstance(path_str, str):
        return False
    # Check for drive path: A:\ or A:/
    if re.match(r"^[A-Z]:[/\\]", path_str, re.IGNORECASE):
        return True
    # Check for UNC path: \\server\share or //server/share
    if path_str.startswith("\\\\") or path_str.startswith("//"):
        return True
    return False


def _extract_filename_from_path(path_str: str) -> str:
    """Extract filename (with pattern) from drive/UNC path."""
    if not _is_drive_or_unc_path(path_str):
        return path_str
    
    # Normalize path separators
    normalized = path_str.replace("\\", "/")
    
    # Remove drive letter prefix (e.g., "Z:/" or "Z:\")
    normalized = re.sub(r"^[A-Z]:/", "", normalized, flags=re.IGNORECASE)
    
    # Remove UNC prefix (e.g., "//server/share/" or "\\server\share\")
    # Keep only the relative path after the share name
    if normalized.startswith("//"):
        # Find the third slash (after //server/share/)
        parts = normalized.split("/")
        if len(parts) >= 4:
            # Reconstruct from the 4th part onwards (skip //server/share)
            normalized = "/".join(parts[3:])
        else:
            # If no path after share, return basename
            normalized = os.path.basename(normalized)
    
    # Extract just the filename (last component)
    filename = os.path.basename(normalized)
    
    # If filename is empty or just a folder name, return the original
    if not filename or filename == normalized:
        return path_str
    
    logging.debug(
        f"{log_tag('NASDB','QFILE')} Extracted filename '{filename}' from path '{path_str}'"
    )
    return filename


def _looks_like_path(value: str) -> bool:
    """Check if a value looks like a file path."""
    if not isinstance(value, str):
        return False
    # Check for drive/UNC path
    if _is_drive_or_unc_path(value):
        return True
    # Check for path separators
    has_sep = ("\\" in value) or ("/" in value)
    # Check for file extension
    has_ext = any(value.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)
    return has_sep or has_ext


class NAS_DB:
    """
    Handles data access from a NAS storage, either directly via
    **a mounted network drive** or **remotely via an SSH connection**.

    Attributes:
        nas_path(Path): The UNC path to the root of the NAS data share.
        nas_mount(Path): The mount point on the local machine (if available, for direct access).
        nas_user(str): The user of the NAS.
        nas_password(str): The password of the NAS.
        default_data_folders: The default data folders to search for data.
        ssh_host(str): The host of the SSH server.
        ssh_port(int): The port of the SSH server.
        ssh_user(str): The user of the SSH server.
        ssh_pkey(Path): The path to the private key file for SSH authentication.
        remote_temp_dir(Path): The temporary path on the remote machine to store scripts.
        ssh_max_retries(int): The number of times to retry the SSH connection.
        ssh_connect_timeout(float): The timeout for establishing the SSH connection.
        dumping_folder(Path): The local directory where HDF5-cached data is stored.
        max_load_size_gb(float): The maximum total size of files to load in a single get_shot_data call (in GB).
        ssh_client(paramiko.SSHClient): The SSH client.
        sftp_client(paramiko.SFTPClient): The SFTP client.
        access_mode(str): The access mode ('local' or 'remote').
        ssh_lock(threading.Lock): The lock for the SSH connection.
        logger(logging.Logger): The logger for the NAS_DB class.
        _file_cache(dict): The cache for file paths for find_files method.
        _is_connected(bool): Whether the connection is established.

    Methods:
        connect: Connects to the NAS.
        disconnect: Disconnects from the NAS.
        find_files: Finds files matching a query across multiple folders.
        get_shot_data: Retrieves data for a given shot number, pattern, or list of files.
        get_data_top: Reads the top N lines of a file.
        _ensure_remote_dir_exists: Ensures a directory exists on the remote server using SFTP.
        _authenticate_nas_remote: Authenticates to the NAS on the remote machine.
        _find_files_remote: Finds files matching a query across multiple folders on the remote machine.
        _get_files_total_size: Calculates the total size of a list of files in bytes.
        _read_shot_file: Reads a shot file from the NAS.
        _dispatch_read: Dispatches the read operation to the correct method based on the file extension.
        _read_scope_csv: Reads a scope CSV file.
        _identify_csv_type: Identifies the type of a CSV file.
        _parse_csv_with_type: Parses a CSV file with the correct type.
        _parse_standard_csv: Parses a standard CSV file.
        _parse_mdo3000pc: Parses a MDO3000pc file.
        _parse_mso58: Parses a MSO58 file.
        _read_fpga_dat: Reads a FPGA DAT file.
        _read_matlab_mat: Reads a MATLAB MAT file.
        _get_data_top_remote: Reads the top N lines of a file on the remote machine.
        _get_data_top_local: Reads the top N lines of a file on the local machine.
        _setup_logger: Sets up the logger for the NAS_DB class.

    Examples:
        ```python
        from .nas_db import NAS_DB
        db = NAS_DB("ifi/config.ini")
        db.connect()
        db.find_files(45821)
        db.get_shot_data(45821)
        db.disconnect()
        ```
    """

    def __init__(self, config_path: str = "ifi/config.ini"):
        """
        Initializes the NAS_DB class.
        Args:
            config_path (str): The path to the config file.
        Raises:
            FileNotFoundError: If the config file is not found.
        """
        # Initialize logger
        self._setup_logger()

        # Check if the config file exists
        if not Path(config_path).exists():
            self.logger.error(f"{log_tag('NASDB','CONFIG')} Config file not found: '{config_path}'")
            raise FileNotFoundError(
                f"{log_tag('NASDB','CONFIG')} Config file not found: '{config_path}'"
            )

        # Read the config file
        config = configparser.ConfigParser()
        config.read(config_path)

        # Read the NAS config
        nas_cfg = config["NAS"]
        self.nas_path = nas_cfg.get("path")
        self.nas_mount = nas_cfg.get("mount_point")
        self.nas_user = nas_cfg.get("user")
        self.nas_password = nas_cfg.get("password")

        # Read default data folders, split by comma, and strip whitespace
        folders_str = nas_cfg.get("data_folders", "")
        self.default_data_folders = [
            f.strip() for f in folders_str.split(",") if f.strip()
        ]

        # SSH Config
        ssh_cfg = config["SSH_NAS"]
        self.ssh_host = ssh_cfg.get("ssh_host")
        self.ssh_port = ssh_cfg.getint("ssh_port")
        self.ssh_user = ssh_cfg.get("ssh_user")
        self.ssh_pkey = Path(ssh_cfg.get("ssh_pkey_path")).expanduser()
        self.remote_temp_dir = ssh_cfg.get(
            "remote_temp_dir", None
        )  # Read optional from config

        # Connection settings
        conn_cfg = config["CONNECTION_SETTINGS"]
        self.ssh_max_retries = conn_cfg.getint("ssh_max_retries", 3)
        self.ssh_connect_timeout = conn_cfg.getfloat("ssh_connect_timeout", 10.0)
        # Maximum number of concurrent SSH commands (default: 2)
        # Lower values reduce SSH channel conflicts but may slow down parallel processing
        # Recommended: 1-2 for single SFTP channel, 3-5 for multiple channels
        self.max_concurrent_ssh_commands = conn_cfg.getint("max_concurrent_ssh_commands", 2)

        # Local Cache Config
        if config.has_section("LOCAL_CACHE"):
            cache_cfg = config["LOCAL_CACHE"]
            self.dumping_folder = cache_cfg.get("dumping_folder", "./cache")
            self.max_load_size_gb = cache_cfg.getfloat("max_load_size_gb", 2.0)
        else:
            self.dumping_folder = "./cache"
            self.max_load_size_gb = 2.0

        self.ssh_client = None
        self.sftp_client = None
        self.access_mode = None  # 'local' or 'remote'
        self._file_cache = {}  # Cache for file paths for find_files method

        self._is_connected = False
        self.ssh_lock = threading.Lock()  # Lock for SSH connection management
        # Semaphore to limit concurrent SSH command executions
        # When set to 1, acts like a Mutex, ensuring sequential execution
        # Higher values allow parallelism but may cause SSH channel conflicts
        self.ssh_command_semaphore = threading.Semaphore(self.max_concurrent_ssh_commands)
        # Optional: Mutex for strict sequential execution when needed
        # This can be used in addition to semaphore for critical sections
        self.ssh_operation_lock = threading.Lock() if self.max_concurrent_ssh_commands == 1 else None

    def _ensure_remote_dir_exists(self, remote_path: str, use_semaphore: bool = True):
        """
        Ensures a directory exists on the remote server using SFTP.

        Args:
            remote_path (str): The path to the remote directory.
            use_semaphore (bool): Whether to use semaphore for thread safety.
                Set to False if already called within a semaphore-protected block to avoid deadlock.

        Returns:
            None

        Raises:
            Exception: If the remote directory cannot be created.
        """
        if self.access_mode != "remote" or not self.sftp_client:
            return

        # If remote_temp_dir is not set in config, create it in the user's home dir.
        if not remote_path:
            # normalize() call should also be protected if use_semaphore=True
            semaphore_context = self.ssh_command_semaphore if use_semaphore else nullcontext()
            with semaphore_context:
                try:
                    home_dir = self.sftp_client.normalize(".")
                    remote_path = os.path.join(home_dir, ".ifi_temp").replace("\\", "/")
                    self.remote_temp_dir = remote_path  # Store for later use
                    self.logger.info(
                        f"{log_tag('NASDB','CHKD')} Remote temp directory not configured, using dynamic path: {remote_path}"
                    )
                except Exception as e:
                    self.logger.error(
                        f"{log_tag('NASDB','CHKD')} Could not determine remote home directory: {e}"
                    )
                    raise

        # Use semaphore to limit concurrent SFTP operations
        semaphore_context = self.ssh_command_semaphore if use_semaphore else nullcontext()
        with semaphore_context:
            try:
                # Check if the path exists and is a directory
                self.sftp_client.stat(remote_path)
            except FileNotFoundError:
                # If it doesn't exist, create it
                self.logger.info(
                    f"{log_tag('NASDB','CHKD')} Remote directory '{remote_path}' not found. Creating it."
                )
                self.sftp_client.mkdir(remote_path)
            except Exception as e:
                self.logger.error(
                    f"{log_tag('NASDB','CHKD')} Could not check or create remote directory '{remote_path}': {e}"
                )
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
                self.logger.info(
                    f"{log_tag('NASDB','CONN')} Local NAS mount found at '{self.nas_mount}'. Using direct access."
                )
                self.access_mode = "local"
                self._is_connected = True
                return True

            self.logger.info(f"{log_tag('NASDB','CONN')} Local NAS mount not found. Attempting SSH connection.")
            self.access_mode = "remote"

            for attempt in range(self.ssh_max_retries):
                try:
                    self.logger.info(f"{log_tag('NASDB','CONN')} SSH connection attempt {attempt + 1}/{self.ssh_max_retries}...")
                    self.ssh_client = paramiko.SSHClient()
                    self.ssh_client.set_missing_host_key_policy(
                        paramiko.AutoAddPolicy()
                    )
                    self.ssh_client.connect(
                        hostname=self.ssh_host,
                        port=self.ssh_port,
                        username=self.ssh_user,
                        key_filename=str(self.ssh_pkey),
                        timeout=self.ssh_connect_timeout,
                    )
                    self.sftp_client = self.ssh_client.open_sftp()
                    self.logger.info(f"{log_tag('NASDB','CONN')} SSH connection to {self.ssh_host} successful.")

                    self._authenticate_nas_remote()
                    self._is_connected = True
                    return True

                except Exception as e:
                    self.logger.error(f"{log_tag('NASDB','CONN')} SSH connection attempt {attempt + 1} failed: {e}")
                    self.disconnect()  # disconnect sets _is_connected to False
                    if attempt < self.ssh_max_retries - 1:
                        self.logger.info(f"{log_tag('NASDB','CONN')} Retrying in 3 seconds...")
                        time.sleep(3)

            self._is_connected = False
            self.logger.error(f"{log_tag('NASDB','CONN')} All SSH connection attempts failed.")
            return False

    def _authenticate_nas_remote(self):
        """
        Runs 'net use' on the remote machine to authenticate with the NAS.
        """
        auth_cmd = f'net use "{str(self.nas_path)}" /user:{self.nas_user} {self.nas_password} /persistent:no'
        self.logger.info(f"{log_tag('NASDB','AUTH')} Authenticating to NAS on remote machine.")
        stdin, stdout, stderr = self.ssh_client.exec_command(auth_cmd)

        exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            err = stderr.read().decode("utf-8", errors="ignore").strip()
            # "The command completed successfully." is not an error.
            if "The command completed successfully" not in err:
                self.logger.warning(f"{log_tag('NASDB','AUTH')} NAS authentication may have failed with exit code {exit_status}: {err}")
            else:
                self.logger.info(f"{log_tag('NASDB','AUTH')} NAS authentication successful.")
        else:
            self.logger.info(f"{log_tag('NASDB','AUTH')} NAS authentication successful.")

    def find_files(
        self,
        query: Union[int, str, List[Union[int, str]]],
        data_folders: List[str] = None,
        add_path: bool = False,
        force_remote: bool = False,
    ) -> List[str]:
        """
        Finds all files matching a query across multiple folders.

        Args:
            query (Union[int, str, List[Union[int, str]]]): The query to search for.
            data_folders (List[str]): The folders to search in.
            add_path (bool): Whether to add the path to the query.
            force_remote (bool): Whether to force the use of remote access.

        Returns:
            List[str]: A list of files that match the query.

        Raises:
            ConnectionError: If the connection to the NAS cannot be established.
            ValueError: If the query is invalid.
            TypeError: If the data_folders is invalid.
            Exception: If an error occurs while searching for files.

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
                raise ConnectionError(f"{log_tag('NASDB','QFILE')} Failed to establish connection to NAS.")

        if data_folders is not None:
            if isinstance(data_folders, str):
                data_folders = [data_folders]
            elif isinstance(data_folders, list):
                for i, folder in enumerate(data_folders):
                    if not isinstance(folder, str):
                        self.logger.warning(
                            f"{log_tag('NASDB','QFILE')} Invalid data folder type: {type(folder)}. Converting to string. {folder}"
                        )
                        data_folders[i] = str(folder)
            else:
                self.logger.warning(
                    f"{log_tag('NASDB','QFILE')} Invalid data folder type: {type(data_folders)}. {data_folders}"
                )
                data_folders = [data_folders]
        else:
            data_folders = self.default_data_folders

        if add_path:
            data_folders = list(set(self.default_data_folders + data_folders))

        cache_key = (str(query), tuple(sorted(data_folders)))
        if cache_key in self._file_cache:
            self.logger.info(f"{log_tag('NASDB','QFILE')} Found file list for {cache_key} in cache.")
            return self._file_cache[cache_key]

        self.logger.info(
            f"{log_tag('NASDB','QFILE')} Searching for files for query: {query} in folders: {data_folders}."
        )

        base_path = (
            str(self.nas_mount) if self.access_mode == "local" else str(self.nas_path)
        )

        # --- Handle full paths directly ---
        # Check if query contains drive/UNC paths
        query_items = query if isinstance(query, list) else [query]
        has_paths = any(
            isinstance(item, str) and _is_drive_or_unc_path(item) for item in query_items
        )
        
        if has_paths:
            # For local access, verify paths exist
            if self.access_mode == "local":
                valid_paths = []
                for item in query_items:
                    if isinstance(item, str) and _is_drive_or_unc_path(item):
                        if Path(item).exists():
                            valid_paths.append(item)
                        else:
                            self.logger.warning(
                                f"{log_tag('NASDB','QFILE')} Path does not exist: {item}"
                            )
                    else:
                        valid_paths.append(item)
                if valid_paths and all(
                    isinstance(p, str) and _is_drive_or_unc_path(p) for p in valid_paths
                ):
                    return valid_paths
            # For remote access, extract filenames and use find_files to resolve
            # This will be handled in the pattern extraction below

        # --- Build search patterns for shot numbers and wildcards ---
        search_patterns = []
        for item in query_items:
            # If it's a drive/UNC path, extract filename first
            if isinstance(item, str) and _is_drive_or_unc_path(item):
                item = _extract_filename_from_path(item)
            
            # Add wildcards for all extensions if it's just a number or simple string
            if isinstance(item, int) or (
                isinstance(item, str) and "*" not in item and "." not in item
            ):
                search_patterns.append(f"{item}*.*")
            else:  # It's already a pattern like "45821_*.csv"
                search_patterns.append(str(item))

        all_files: Set[str] = set()
        if self.access_mode == "local":
            for folder in data_folders:
                for pattern in search_patterns:
                    search_path = Path(base_path) / folder / "**" / pattern
                    found = glob.glob(str(search_path), recursive=True)
                    all_files.update(found)
        else:  # remote access
            all_found_paths = self._find_files_remote(data_folders, search_patterns)
            all_files.update(all_found_paths)

        sorted_files = sorted(list(all_files))

        # --- Filter by allowed extensions ---
        filtered_files = [
            f for f in sorted_files if Path(f).suffix.lower() in ALLOWED_EXTENSIONS
        ]
        if len(filtered_files) < len(sorted_files):
            self.logger.info(
                f"{log_tag('NASDB','QFILE')} Filtered file list from {len(sorted_files)} to {len(filtered_files)} "
                f"based on allowed extensions: {ALLOWED_EXTENSIONS}"
            )

        # --- Normalize paths to use forward slashes for consistency ---
        normalized_files = [f.replace("\\", "/") for f in filtered_files]

        if normalized_files:
            self.logger.info(f"{log_tag('NASDB','QFILE')} Found {len(normalized_files)} files. Caching result.")
            self._file_cache[cache_key] = normalized_files
        else:
            self.logger.warning(
                f"{log_tag('NASDB','QFILE')} No files with allowed extensions found for query '{query}' in {data_folders}. Caching empty list."
            )
            self._file_cache[cache_key] = []

        return normalized_files

    def _find_files_remote(
        self, data_folders: List[str], patterns: List[str]
    ) -> List[str]:
        """Executes a single remote script to find files across multiple folders and patterns."""
        patterns_str = " ".join(patterns)

        # Combine all search folders into a single semicolon-separated string
        search_paths = [
            os.path.join(self.nas_path, folder).replace("\\", "/")
            for folder in data_folders
        ]
        search_paths_str = ";".join(search_paths)

        # Generate unique filename to avoid collisions in parallel execution
        script_filename = _generate_unique_script_name(f"list_{int(time.time())}")
        remote_script_path = os.path.join(
            self.remote_temp_dir, script_filename
        ).replace("\\", "/")

        # Use semaphore to limit concurrent SSH command executions
        # This allows up to max_concurrent_ssh_commands commands to run simultaneously
        # while preventing too many concurrent connections that could cause issues
        # Optional mutex lock ensures strict sequential execution queue when max_concurrent_ssh_commands == 1
        # When semaphore value is 1, it acts like a mutex, creating a strict queue
        mutex_context = self.ssh_operation_lock if self.ssh_operation_lock else nullcontext()
        with mutex_context:
            self.logger.debug(
                f"{log_tag('NASDB','QFILE')} Waiting for SSH operation queue (max concurrent: {self.max_concurrent_ssh_commands})"
            )
            with self.ssh_command_semaphore:
                self.logger.debug(
                    f"{log_tag('NASDB','QFILE')} SSH semaphore acquired. Starting remote file search for patterns: {patterns}"
                )
                # Ensure the remote temp directory exists before trying to write to it
                # Note: use_semaphore=False because we're already in a semaphore block
                self._ensure_remote_dir_exists(self.remote_temp_dir, use_semaphore=False)
                try:
                    with self.sftp_client.open(remote_script_path, "w") as f:
                        f.write(REMOTE_LIST_SCRIPT)
                except Exception as e:
                    self.logger.error(f"{log_tag('NASDB','QFILE')} Failed to write remote list script: {e}")
                    return []

                cmd = f'python "{remote_script_path}" "{search_paths_str}" "{patterns_str}"'
                
                self.logger.debug(
                    f"{log_tag('NASDB','QFILE')} Executing remote command: python script"
                )
                stdin, stdout, stderr = self.ssh_client.exec_command(cmd)
                
                self.logger.debug(
                    f"{log_tag('NASDB','QFILE')} Reading command output..."
                )
                files = stdout.read().decode("utf-8").strip().splitlines()
                err_output = stderr.read().decode("utf-8", errors="ignore").strip()

                if err_output:
                    self.logger.error(f"{log_tag('NASDB','QFILE')} Remote list script error: {err_output}")

                self.logger.debug(
                    f"{log_tag('NASDB','QFILE')} SSH semaphore released. Found {len(files)} files."
                )
                return files

    def _get_files_total_size(self, file_list: List[str]) -> int:
        """Calculates the total size of a list of files in bytes."""
        total_size = 0
        if not file_list:
            return 0

        self.logger.info(f"{log_tag('NASDB','QBYTE')} Calculating total size of {len(file_list)} files...")

        try:
            if self.access_mode == "local":
                for file_path in file_list:
                    if Path(file_path).exists():
                        total_size += Path(file_path).stat().st_size
            else:  # remote
                for file_path in file_list:
                    try:
                        # Use semaphore to limit concurrent SFTP operations
                        # This prevents too many simultaneous stat() calls that could cause issues
                        with self.ssh_command_semaphore:
                            total_size += self.sftp_client.stat(file_path).st_size
                    except FileNotFoundError:
                        self.logger.warning(
                            f"{log_tag('NASDB','QBYTE')} Remote file not found during size calculation: {file_path}"
                        )
                        continue

            return total_size
        except Exception as e:
            self.logger.error(f"{log_tag('NASDB','QBYTE')} Error calculating file sizes: {e}")
            return -1  # Return -1 to indicate an error

    def _is_valid_data(self, df: pd.DataFrame) -> bool:
        """
        Check if DataFrame contains valid data (no NaN, None, inf, or empty).
        
        Args:
            df: DataFrame to validate
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        if df is None:
            return False
        
        if df.empty:
            return False
        
        # Check for NaN values
        if df.isnull().any().any():
            return False
        
        # Check for inf values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            if np.isinf(df[numeric_cols]).any().any():
                return False
        
        # Check if DataFrame has any actual data (not just structure)
        if df.shape[0] == 0 or df.shape[1] == 0:
            return False
        
        return True

    def _load_signals_from_results(
        self, 
        shot_num: int, 
        results_dir: Path,
        allowed_frequencies: List[float] = None
    ) -> Dict[str, pd.DataFrame] | None:
        """
        Load signals from results directory if available and valid.
        
        Args:
            shot_num: Shot number
            results_dir: Results base directory path (e.g., project_root / "ifi" / "results")
            allowed_frequencies: Optional list of allowed frequencies in GHz (e.g., [94.0, 280.0]).
                                If None, all frequencies are loaded.
            
        Returns:
            Dictionary mapping signal names to DataFrames, or None if not available/invalid
        """
        results_file = results_dir / str(shot_num) / f"{shot_num}.h5"
        
        if not results_file.exists():
            return None
        
        try:
            with h5py.File(results_file, "r") as hf:
                if "signals" not in hf:
                    return None
                
                signals_group = hf["signals"]
                
                # Check if signals group is empty
                if signals_group.attrs.get("empty", False):
                    return None
                
                signals_dict = {}
                for signal_name in signals_group.keys():
                    # Extract frequency from signal name (e.g., 'freq_94.0_GHz' -> 94.0)
                    freq_match = re.search(r"freq_(\d+\.?\d*)_GHz", signal_name)
                    if freq_match:
                        signal_freq = float(freq_match.group(1))
                        
                        # Map to standard frequency group (93-95 -> 94, 275-285 -> 280)
                        if 93.0 <= signal_freq <= 95.0:
                            group_freq = 94.0
                        elif 275.0 <= signal_freq <= 285.0:
                            group_freq = 280.0
                        else:
                            group_freq = signal_freq
                        
                        # Filter by allowed frequencies if specified
                        if allowed_frequencies is not None:
                            if group_freq not in allowed_frequencies:
                                self.logger.debug(
                                    f"{log_tag('NASDB','RSLT')} Skipping signal '{signal_name}' "
                                    f"(frequency {group_freq} GHz not in allowed list: {allowed_frequencies})"
                                )
                                continue
                    
                    signal_group = signals_group[signal_name]
                    signal_data = {}
                    
                    for col_name in signal_group.keys():
                        signal_data[col_name] = signal_group[col_name][:]
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(signal_data)
                    
                    # Validate data
                    if not self._is_valid_data(df):
                        self.logger.warning(
                            f"{log_tag('NASDB','RSLT')} Signal '{signal_name}' in results has invalid data. Skipping."
                        )
                        continue
                    
                    signals_dict[signal_name] = df
                    self.logger.info(
                        f"{log_tag('NASDB','RSLT')} Loaded signal '{signal_name}' from results with shape {df.shape}"
                    )
                
                if signals_dict:
                    return signals_dict
                else:
                    return None
                    
        except Exception as e:
            self.logger.warning(
                f"{log_tag('NASDB','RSLT')} Error loading signals from results file '{results_file}': {e}"
            )
            return None

    def _convert_results_signals_to_nas_format(
        self, 
        results_signals: Dict[str, pd.DataFrame],
        target_files: List[str],
        shot_num: int
    ) -> Dict[str, pd.DataFrame]:
        """
        Convert results signals format to NAS_DB format (file basename as key).
        
        Uses get_interferometry_params() to determine which files belong to which frequency,
        then matches results signals (organized by frequency) to the corresponding files.
        
        Args:
            results_signals: Dictionary from results (keys: signal names like 'freq_94.0_GHz')
            target_files: List of target file paths to match
            shot_num: Shot number for interferometry parameter lookup
            
        Returns:
            Dictionary with file basenames as keys
        """
        from ..analysis.phi2ne import get_interferometry_params
        
        nas_format_dict = {}
        
        # First, determine which files belong to which frequency using get_interferometry_params
        file_freq_map = {}  # Maps file basename -> frequency group (94.0 or 280.0)
        for file_path in target_files:
            basename = Path(file_path).name
            try:
                params = get_interferometry_params(shot_num, basename)
                freq_ghz = params.get("freq_ghz", 94.0)
                
                # Map to standard frequency group
                if 93.0 <= freq_ghz <= 95.0:
                    group_freq = 94.0
                elif 275.0 <= freq_ghz <= 285.0:
                    group_freq = 280.0
                else:
                    group_freq = freq_ghz
                
                file_freq_map[basename] = group_freq
                self.logger.debug(
                    f"{log_tag('NASDB','RSLT')} File '{basename}' mapped to frequency group {group_freq} GHz"
                )
            except Exception as e:
                self.logger.warning(
                    f"{log_tag('NASDB','RSLT')} Could not determine frequency for file '{basename}': {e}"
                )
                continue
        
        # Now match results signals to files based on frequency
        for signal_name, df in results_signals.items():
            # Extract frequency from signal name (e.g., 'freq_94.0_GHz' -> 94.0)
            freq_match = re.search(r"freq_(\d+\.?\d*)_GHz", signal_name)
            if not freq_match:
                self.logger.warning(
                    f"{log_tag('NASDB','RSLT')} Could not extract frequency from signal name '{signal_name}'"
                )
                continue
            
            signal_freq = float(freq_match.group(1))
            
            # Map to standard frequency group
            if 93.0 <= signal_freq <= 95.0:
                group_freq = 94.0
            elif 275.0 <= signal_freq <= 285.0:
                group_freq = 280.0
            else:
                group_freq = signal_freq
            
            # Find files that match this frequency
            matching_files = [
                basename for basename, freq in file_freq_map.items()
                if abs(freq - group_freq) < 0.1
            ]
            
            if not matching_files:
                self.logger.warning(
                    f"{log_tag('NASDB','RSLT')} No matching files found for signal '{signal_name}' "
                    f"(frequency group {group_freq} GHz)"
                )
                continue
            
            # For 280GHz: typically one _ALL file
            # For 94GHz: multiple files (_056, _789, etc.) - combine them
            if group_freq == 280.0:
                # Find _ALL file
                all_file = next((f for f in matching_files if "_ALL" in f), None)
                if all_file:
                    nas_format_dict[all_file] = df
                    self.logger.info(
                        f"{log_tag('NASDB','RSLT')} Matched results signal '{signal_name}' to file '{all_file}'"
                    )
                else:
                    self.logger.warning(
                        f"{log_tag('NASDB','RSLT')} No _ALL file found for 280GHz signal '{signal_name}'"
                    )
            else:  # 94GHz - multiple files
                # For 94GHz, the results signal contains combined data from multiple files
                # We need to assign it to all matching files, or use the first one as representative
                # Since results contain combined data, we'll use the first matching file as key
                # and log that it represents multiple files
                first_file = matching_files[0]
                nas_format_dict[first_file] = df
                self.logger.info(
                    f"{log_tag('NASDB','RSLT')} Matched results signal '{signal_name}' to file '{first_file}' "
                    f"(represents {len(matching_files)} files: {matching_files})"
                )
        
        return nas_format_dict

    def get_shot_data(
        self,
        query: Union[int, str, List[Union[int, str]]],
        data_folders: Union[list, str] = None,
        add_path: bool = False,
        force_remote: bool = False,
        allowed_frequencies: List[float] = None,
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        """
        Retrieves data for a given shot number, pattern, or list of files.
        Caches each file to a dedicated HDF5 file based on its shot number.

        Args:
            query (Union[int, str, List[Union[int, str]]]): The query to search for.
            data_folders (Union[list, str]): The folders to search in. Default is None.
            add_path (bool): Whether to add the path to the query. Default is False.
            force_remote (bool): If True, bypasses the local cache and fetches from the NAS. Default is False.
            **kwargs: Additional arguments passed to the data parsing functions (e.g., skiprows). Default is None.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary mapping each successfully read filename to its DataFrame.

        Raises:
            MemoryError: If the total size of the files to fetch exceeds the configured limit.
        """
        if not self._is_connected:
            if not self.connect():
                raise ConnectionError(f"{log_tag('NASDB','QSHOT')} Failed to establish connection to NAS.")

        query_is_path = False
        if isinstance(query, str) and _looks_like_path(query):
            query_is_path = True
        elif isinstance(query, list) and query and all(_looks_like_path(q) for q in query):
            query_is_path = True

        if query_is_path:
            if isinstance(query, list):
                _check_target_files = query
            else:
                _check_target_files = [query]
            
            # For drive/UNC paths, use find_files to resolve them properly
            # This ensures paths are converted to accessible formats for remote access
            target_files = self.find_files(
                _check_target_files, data_folders, add_path, force_remote, **kwargs
            )
            self.logger.info(
                f"{log_tag('NASDB','QSHOT')} Using provided file path(s) directly: {target_files}"
            )
        else:
            # --- Find all target files on the NAS ---
            target_files = self.find_files(
                query, data_folders, add_path, force_remote, **kwargs
            )
            if not target_files:
                self.logger.warning(f"{log_tag('NASDB','QSHOT')} No files found on NAS for query: {query}")
                return {}

        data_dict: Dict[str, pd.DataFrame] = {}
        files_to_fetch = []

        if force_remote:
            files_to_fetch = target_files
        else:
            # --- Check results directory first (highest priority) ---
            # Extract shot numbers from target files
            shot_numbers_in_query = set()
            for file_path in target_files:
                basename = _extract_filename_from_path(file_path)
                match = re.match(r"(\d+)", basename)
                if match:
                    shot_numbers_in_query.add(int(match.group(1)))
            
            # Try to load from results for each shot number
            project_root = get_project_root()
            results_base_dir = project_root / "ifi" / "results"
            
            results_signals_loaded = {}
            for shot_num in shot_numbers_in_query:
                results_signals = self._load_signals_from_results(
                    shot_num, results_base_dir, allowed_frequencies=allowed_frequencies
                )
                if results_signals:
                    results_signals_loaded[shot_num] = results_signals
                    self.logger.info(
                        f"{log_tag('NASDB','RSLT')} Found valid signals in results for shot {shot_num}"
                    )
            
            # Convert results signals to NAS format and add to data_dict
            if results_signals_loaded:
                for shot_num, signals in results_signals_loaded.items():
                    # Match results signals to target files for this shot
                    shot_target_files = [
                        f for f in target_files
                        if re.match(r"(\d+)", _extract_filename_from_path(f)) and
                        int(re.match(r"(\d+)", _extract_filename_from_path(f)).group(1)) == shot_num
                    ]
                    converted_signals = self._convert_results_signals_to_nas_format(
                        signals, shot_target_files, shot_num
                    )
                    data_dict.update(converted_signals)
            
            # Determine which files still need to be fetched
            # Files that were successfully loaded from results don't need to be fetched
            for file_path in target_files:
                basename = _extract_filename_from_path(file_path)
                if basename not in data_dict:
                    files_to_fetch.append(file_path)
                else:
                    self.logger.info(
                        f"{log_tag('NASDB','RSLT')} Skipping '{basename}' - already loaded from results"
                    )
            
            # --- Check cache for remaining files ---
            remaining_files_to_fetch = []
            for file_path in files_to_fetch:
                basename = _extract_filename_from_path(file_path)
                # Try to extract shot number from filename (e.g., "45821_056.csv" -> "45821")
                match = re.match(r"(\d+)", basename)
                shot_num_for_cache = int(match.group(1)) if match else None

                if shot_num_for_cache is None:
                    self.logger.warning(
                        f"{log_tag('NASDB','QSHOT')} Could not determine shot number for '{basename}'. Will not use cache."
                    )
                    remaining_files_to_fetch.append(file_path)
                    continue

                cache_dir = Path(self.dumping_folder) / str(shot_num_for_cache)
                cache_file = cache_dir / f"{shot_num_for_cache}.h5"

                # --- Start Diagnostic Logging ---
                self.logger.info(f"{log_tag('NASDB','CACHE')} For file: '{basename}'")
                self.logger.info(f"{log_tag('NASDB','CACHE')} Checking for cache file at: '{cache_file.absolute()}'")

                if not cache_file.exists():
                    self.logger.warning(f"{log_tag('NASDB','CACHE')} -> Cache file NOT FOUND.")
                    remaining_files_to_fetch.append(file_path)
                    continue

                self.logger.info(f"{log_tag('NASDB','CACHE')} -> Cache file FOUND.")
                # --- End Diagnostic Logging ---

                # Sanitize the basename to be a valid HDF5 key, matching the writing logic.
                key = re.sub(r"[^a-zA-Z0-9_]", "_", basename)
                if key and not key[0].isalpha() and not key.startswith("_"):
                    key = "_" + key

                # --- Start Diagnostic Logging ---
                self.logger.info(f"{log_tag('NASDB','CACHE')} Looking for key: '{key}'")
                # --- End Diagnostic Logging ---

                try:
                    with h5py.File(cache_file, "r") as f:
                        if key in f:
                            self.logger.info(f"{log_tag('NASDB','CACHE')} -> Key FOUND in cache. Loading from HDF5.")
                            self.logger.info(
                                f"{log_tag('NASDB','CACHE')} Found '{basename}' in cache: {cache_file}"
                            )
                            df = pd.read_hdf(cache_file, key)

                            # Restore metadata if available
                            metadata_key = f"{key}_metadata"
                            if metadata_key in f:
                                self.logger.info(
                                    f"{log_tag('NASDB','CACHE')} Restoring metadata for '{basename}' from key '{metadata_key}'"
                                )
                                metadata_series = pd.read_hdf(cache_file, metadata_key)
                                metadata_dict = metadata_series.to_dict()

                                # Convert JSON string to dictionary
                                import json

                                for k, v in metadata_dict.items():
                                    if k == "metadata" and isinstance(v, str):
                                        try:
                                            df.attrs[k] = json.loads(v)
                                        except:  # noqa: E722
                                            df.attrs[k] = v
                                    else:
                                        df.attrs[k] = v

                                self.logger.info(f"{log_tag('NASDB','CACHE')} Restored metadata: {df.attrs}")

                            # Use basename as key for consistency (avoids network path issues)
                            data_dict[basename] = df
                        else:
                            self.logger.warning(f"{log_tag('NASDB','CACHE')} -> Key NOT FOUND in cache file. Re-fetching.")
                            remaining_files_to_fetch.append(file_path)
                except Exception as e:
                    self.logger.error(
                        f"{log_tag('NASDB','CACHE')} Error reading cache file '{cache_file}': {e}. Will refetch."
                    )
                    remaining_files_to_fetch.append(file_path)
            
            # Update files_to_fetch with remaining files
            files_to_fetch = remaining_files_to_fetch

        if not files_to_fetch:
            self.logger.info(f"{log_tag('NASDB','CACHE')} All required files were found in the cache.")
            return data_dict

        # --- Memory Check ---
        if self.max_load_size_gb > 0:
            total_size_bytes = self._get_files_total_size(files_to_fetch)
            total_size_gb = total_size_bytes / (1024**3)

            if total_size_bytes < 0:
                self.logger.warning(
                    f"{log_tag('NASDB','QSHOT')} Could not determine total size of files to fetch. Proceeding with caution."
                )
            elif total_size_gb > self.max_load_size_gb:
                raise MemoryError(
                    f"{log_tag('NASDB','QSHOT')} Total size of files to fetch ({total_size_gb:.2f} GB) exceeds the configured limit "
                    f"of {self.max_load_size_gb:.2f} GB. To override, increase 'max_load_size_gb' in "
                    f"config.ini or use a more specific query."
                )
            else:
                self.logger.info(
                    f"{log_tag('NASDB','QSHOT')} Total size of files to fetch: {total_size_gb:.2f} GB. (Limit: {self.max_load_size_gb:.2f} GB)"
                )

        # --- Fetching and Caching Logic ---
        self.logger.info(f"{log_tag('NASDB','QSHOT')} Fetching {len(files_to_fetch)} files from NAS...")
        for file_path in files_to_fetch:
            df = self._read_shot_file(file_path, **kwargs)
            if df is not None:
                # Extract basename for consistent key usage (avoids network path issues)
                basename = Path(file_path).name
                # Use basename as key for consistency
                data_dict[basename] = df

                # --- Cache the newly fetched file ---
                match = re.match(r"(\d+)", basename)
                shot_num_for_cache = int(match.group(1)) if match else None

                if shot_num_for_cache is not None:
                    cache_dir = Path(self.dumping_folder) / str(shot_num_for_cache)
                    cache_file = cache_dir / f"{shot_num_for_cache}.h5"

                    cache_dir.mkdir(parents=True, exist_ok=True)
                    # Sanitize the basename to be a valid HDF5 key
                    key = re.sub(r"[^a-zA-Z0-9_]", "_", basename)
                    if key and not key[0].isalpha() and not key.startswith("_"):
                        key = "_" + key
                    self.logger.info(
                        f"{log_tag('NASDB','CACHE')} Caching '{basename}' to '{cache_file}' with key '{key}'"
                    )
                    # Use keyword arguments for to_hdf for future compatibility with pandas 3.0
                    df.to_hdf(
                        path_or_buf=cache_file,
                        key=key,
                        mode="a",
                        format="table",
                        complevel=5,
                        complib="zlib",
                    )

                    # Save metadata separately to preserve attrs
                    if hasattr(df, "attrs") and df.attrs:
                        metadata_key = f"{key}_metadata"
                        # Save metadata as a string
                        metadata_dict = {}
                        for k, v in df.attrs.items():
                            if isinstance(v, dict):
                                # Convert nested dictionaries to JSON string
                                import json

                                metadata_dict[k] = json.dumps(v)
                            else:
                                metadata_dict[k] = str(v)

                        metadata_series = pd.Series(metadata_dict)
                        metadata_series.to_hdf(
                            path_or_buf=cache_file,
                            key=metadata_key,
                            mode="a",
                            format="table",
                            complevel=5,
                            complib="zlib",
                        )
                        self.logger.info(
                            f"{log_tag('NASDB','CACHE')} Cached metadata for '{basename}' with key '{metadata_key}'"
                        )
                else:
                    self.logger.warning(
                        f"{log_tag('NASDB','CACHE')} Could not determine shot number for '{basename}'. Skipping cache."
                    )

        # Sort the final dictionary by key (filename) for consistent order
        return dict(sorted(data_dict.items()))

    def _read_shot_file(self, file_path: str, **kwargs) -> pd.DataFrame | None:
        """
        Master parser that dispatches to the correct reader based on file extension.
        This method is a thread-safe wrapper for remote operations.

        Args:
            file_path (str): The path to the file.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.DataFrame | None: The parsed data. None if an error occurs.
        """
        if self.access_mode == "remote":
            # Use semaphore instead of lock to allow limited parallelism
            # This allows up to max_concurrent_ssh_commands file reads simultaneously
            # while preventing too many concurrent SFTP operations that could cause channel issues
            with self.ssh_command_semaphore:
                return self._dispatch_read(file_path, **kwargs)
        else:
            # No lock needed for local access
            return self._dispatch_read(file_path, **kwargs)

    def _dispatch_read(self, file_path: str, **kwargs) -> pd.DataFrame | None:
        """
        Internal dispatcher for reading files. Assumes any necessary locks are held.

        Args:
            file_path (str): The path to the file.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.DataFrame | None: The parsed data. None if an error occurs.
        """
        ext = Path(file_path).suffix.lower()

        if ext == ".csv":
            return self._read_scope_csv(file_path, **kwargs)
        elif ext == ".dat":
            return self._read_fpga_dat(file_path, **kwargs)
        elif ext == ".mat":
            return self._read_matlab_mat(file_path, **kwargs)
        else:
            self.logger.warning(
                f"{log_tag('NASDB','QEXTN')} Unsupported file extension '{ext}' for file: {file_path}"
            )
            return None

    def _read_scope_csv(self, file_path: str, **kwargs) -> pd.DataFrame | None:
        """
        Reads a CSV file by first identifying its type from the header.
        Thread-safe for network drives with file-level locking.

        Args:
            file_path (str): The path to the file.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.DataFrame | None: The parsed data. None if an error occurs.

        Raises:
            read_error: If an error occurs while reading the file.
        """
        self.logger.info(f"{log_tag('NASDB','QCSV')} Reading CSV file: {file_path}")

        # Get header without any locks for full parallel processing
        # Note: _get_data_top_remote is called with use_semaphore=False because
        # _read_shot_file already protects this call with ssh_command_semaphore
        header_text = None
        if self.access_mode == "local":
            header_text = self._get_data_top_local(file_path, lines=40)
        else:  # remote
            header_text = self._get_data_top_remote(file_path, lines=40, use_semaphore=False)

        if not header_text:
            self.logger.error(f"{log_tag('NASDB','QCSV')} Could not read header of {file_path}")
            return None

        header_content = header_text.splitlines()
        csv_type = self._identify_csv_type(header_content)
        self.logger.info(
            f"{log_tag('NASDB','QCSV')} Identified CSV type for {Path(file_path).name} as: {csv_type}"
        )

        # All files run in parallel - no locks
        try:
            df = self._parse_csv_with_type(
                csv_type, file_path, header_content, **kwargs
            )

            if df is not None:
                df.attrs["source_file_type"] = "csv"
                df.attrs["source_file_format"] = csv_type
                self.logger.info(f"{log_tag('NASDB','QCSV')} Successfully parsed file: {file_path}")
                return df  # Success, return immediately
        except Exception as read_error:
            self.logger.error(
                f"{log_tag('NASDB','QCSV')} Failed to parse file '{file_path}'. Error: {read_error}"
            )
            return None

    def _parse_csv_with_type(
        self, csv_type: str, file_path: str, header_content: list, **kwargs
    ) -> pd.DataFrame:
        """
        Parse CSV file based on its identified type.

        Args:
            csv_type (str): The type of the CSV file.
            file_path (str): The path to the file.
            header_content (list[str]): The header content of the file.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.DataFrame: The parsed data.
        """
        if csv_type == "MDO3000pc":
            return self._parse_mdo3000pc(file_path, header_content, **kwargs)
        elif csv_type == "MSO58":
            return self._parse_mso58(file_path, header_content, **kwargs)
        elif csv_type in ["MDO3000orig", "MDO3000fetch", "ETC"]:
            return self._parse_standard_csv(
                file_path, header_content, csv_type, **kwargs
            )
        else:
            self.logger.error(f"{log_tag('NASDB','PCSV')} Unknown CSV type '{csv_type}' for {file_path}")
            return None

    def _identify_csv_type(self, header_content: list[str]) -> str:
        """
        Identifies the CSV format based on the first 40 lines of content.

        Args:
            header_content (list[str]): The header content of the file.

        Returns:
            str: The type of the CSV file. "UNKNOWN" if the header content cannot be parsed.
        """
        # Placeholder for the complex identification logic
        # For now, returning ETC as a default
        try:
            lines = [line.strip().split(",") for line in header_content]
            if not lines:
                return "UNKNOWN"

            # MDO3000 / MSO58 Check
            if "model" in lines[0][0].lower():
                if len(lines[0]) > 1:
                    model_name = lines[0][1]
                    if "MDO3" in model_name:
                        if len(lines) > 1 and "firmware version" in lines[1][0].lower():
                            return "MDO3000orig"
                        else:
                            return "MDO3000fetch"
                    elif "MSO5" in model_name:
                        return "MSO58"

            # MDO3000pc Check
            if len(lines) > 16 and len(lines[16]) > 1 and "MDO3" in lines[16][1]:
                return "MDO3000pc"

        except IndexError:
            # File is likely not in a recognized format, treat as ETC
            pass

        return "ETC"

    def _parse_mdo3000pc(
        self, file_path: str, header_content: list[str], **kwargs
    ) -> pd.DataFrame | None:
        """
        Parses a MDO3000pc file with a fixed header length and no data header row.

        Args:
            file_path (str): The path to the file.
            header_content (list[str]): The header content of the file.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.DataFrame | None: The parsed data. None if an error occurs.

        Raises:
            ValueError: If the header content cannot be parsed.
            IndexError: If the header content cannot be parsed.
            Exception: If an error occurs while parsing the file.
        """
        self.logger.info(f"{log_tag('NASDB','P3KPC')} Parsing as MDO3000pc: {file_path}")

        metadata = {}
        source_line_content = []
        source_line_index = -1

        # 1. Find metadata and the 'Source' line
        for i, line in enumerate(header_content):
            parts = line.strip().split(",")
            if len(parts) > 1:
                key = parts[0].strip()
                val = parts[1].strip()
                if "Record Length" in key:
                    metadata["record_length"] = int(val)
                elif "Sample Interval" in key:
                    metadata["time_resolution"] = float(val)

            if "Source" in line:
                source_line_content = parts
                source_line_index = i

        if source_line_index == -1:
            self.logger.error(
                f"{log_tag('NASDB','P3KPC')} Could not find 'Source' line in header for MDO3000pc file: {file_path}"
            )
            return None

        self.logger.info(
            f"{log_tag('NASDB','P3KPC')} MDO3000pc 'Source' line content: {source_line_content}"
        )  # Debugging line

        # 2. Determine column names and data column indices from 'Source' line
        ch_names = []
        data_col_indices = []
        for i, part in enumerate(source_line_content):
            if "Source" in part and i + 1 < len(source_line_content):
                ch_names.append(source_line_content[i + 1].strip())

            # Make parsing more robust - try converting to float
            try:
                float(part.strip())
                data_col_indices.append(i)
            except ValueError:
                continue

        # This is interpreted as taking the 1st, 2nd, 4th... (i.e., 1st and even-indexed) numeric columns
        final_data_indices = [data_col_indices[0]]
        final_data_indices.extend(
            [data_col_indices[i] for i in range(len(data_col_indices)) if i % 2 == 1]
        )

        if not final_data_indices:
            self.logger.error(
                f"{log_tag('NASDB','P3KPC')} Could not extract data column indices for MDO3000pc file: {file_path}"
            )
            return None

        # 3. Read the file with pandas, using the determined indices
        read_target = file_path
        temp_file = None
        try:
            # The actual data starts after the header block (assumed to be 17 lines)
            if self.access_mode == "remote":
                self.logger.info(
                    f"{log_tag('NASDB','P3KPC')} Downloading remote file to temp location: {file_path}"
                )
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                self.sftp_client.get(file_path, temp_file.name)
                read_target = temp_file.name
            else:
                # Use pathlib.Path for safe local file handling
                read_target = Path(read_target)

            # Prefer C engine; on failure, fallback to python engine (without low_memory)
            try:
                p = ensure_str_path(read_target)
                # Use file handle to avoid Windows path quirks under threaded IO
                with open(p, "rb") as fh:
                    df = pd.read_csv(
                        fh,
                        skiprows=17,
                        header=None,
                        usecols=final_data_indices,
                        low_memory=False,
                        on_bad_lines="warn",
                        encoding_errors="ignore",
                        memory_map=False,
                    )
            except Exception as e_c:
                self.logger.warning(
                    f"{log_tag('NASDB','P3KPC')} C engine parse failed for MDO3000pc: {e_c}. Retrying with python engine."
                )
                p = ensure_str_path(read_target)
                with open(p, "rb") as fh:
                    df = pd.read_csv(
                        fh,
                        skiprows=17,
                        header=None,
                        usecols=final_data_indices,
                        on_bad_lines="warn",
                        encoding_errors="ignore",
                        engine="python",
                    )

            # Standardize column names: TIME for first column, CH0, CH1, CH2... for the rest
            standardized_cols = ["TIME"] + [
                f"CH{i}" for i in range(len(df.columns) - 1)
            ]
            df.columns = standardized_cols

            df.attrs["metadata"] = metadata
            return df

        except Exception as e:
            self.logger.error(
                f"{log_tag('NASDB','P3KPC')} Pandas failed to parse MDO3000pc file {file_path}. Error: {e}"
            )
            return None
        finally:
            if temp_file:
                temp_file.close()
                os.unlink(temp_file.name)

    def _parse_mso58(
        self, file_path: str, header_content: list[str], **kwargs
    ) -> pd.DataFrame | None:
        """
        Parses data from a Tektronix MSO58 series oscilloscope CSV file.
        This format has a fixed header length and no data header row.

        Args:
            file_path (str): The path to the file.
            header_content (list[str]): The header content of the file.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.DataFrame | None: The parsed data. None if an error occurs.

        Raises:
            ValueError: If the header content cannot be parsed.
            IndexError: If the header content cannot be parsed.
            Exception: If an error occurs while parsing the file.
        """
        self.logger.info(f"{log_tag('NASDB','PMSO5')} Parsing as MSO58: {file_path}")

        metadata = {}
        header_row_index = -1

        # 1. Dynamically find the header row, similar to _parse_standard_csv
        for i, line in enumerate(header_content):
            # Parse metadata from any line
            try:
                parts = line.strip().split(",")
                if len(parts) > 1:
                    key = parts[0].strip()
                    val = parts[1].strip()
                    if "Record Length" in key:
                        metadata["record_length"] = int(val)
                    elif "Sample Interval" in key:
                        metadata["time_resolution"] = float(val)
            except (ValueError, IndexError):
                pass  # Ignore lines that can't be parsed for metadata

            # Find the actual header row
            if "TIME" in line.upper() and "CH" in line.upper():
                header_row_index = i
                self.logger.info(
                    f"{log_tag('NASDB','PMSO5')} Dynamically detected MSO58 header at line: {header_row_index}"
                )
                break

        if header_row_index == -1:
            self.logger.error(
                f"{log_tag('NASDB','PMSO5')} Could not dynamically determine the header row for MSO58 file: {file_path}"
            )
            return None

        read_target = file_path
        temp_file = None
        try:
            if self.access_mode == "remote":
                self.logger.info(
                    f"{log_tag('NASDB','PMSO5')} Downloading remote file to temp location: {file_path}"
                )
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                self.sftp_client.get(file_path, temp_file.name)
                read_target = temp_file.name
            else:
                # Use pathlib.Path for safe local file handling
                read_target = Path(read_target)

            df = pd.read_csv(
                read_target,
                skiprows=header_row_index,
                header=0,  # Use the found line as the header
                encoding_errors="ignore",
                engine="python",
                skipfooter=0,
            )

            # Standardize column names: TIME for first column, CH0, CH1, CH2... for the rest
            standardized_cols = ["TIME"] + [
                f"CH{i}" for i in range(len(df.columns) - 1)
            ]
            df.columns = standardized_cols

            # Verification log
            actual_len = len(df)
            expected_len = metadata.get("record_length")
            self.logger.info(
                f"{log_tag('NASDB','PMSO5')} File: {Path(file_path).name}, Record Length from header: {expected_len}, Actual rows read: {actual_len}"
            )
            if expected_len is not None and expected_len != actual_len:
                self.logger.warning(
                    f"{log_tag('NASDB','PMSO5')} RECORD LENGTH MISMATCH for {Path(file_path).name}! Header: {expected_len}, Actual: {actual_len}"
                )

            df.attrs["metadata"] = metadata
            return df
        except Exception as e:
            self.logger.error(
                f"{log_tag('NASDB','PMSO5')} Pandas failed to parse MSO58 file {file_path}. Error: {e}"
            )
            return None
        finally:
            if temp_file:
                temp_file.close()
                os.unlink(temp_file.name)

    def _parse_standard_csv(
        self, file_path: str, header_content: list[str], csv_type: str, **kwargs
    ) -> pd.DataFrame | None:
        """
        Parses a standard CSV file with a fixed header length and no data header row.

        Args:
            file_path (str): The path to the file.
            header_content (list[str]): The header content of the file.
            csv_type (str): The type of the CSV file.

        Returns:
            pd.DataFrame | None: The parsed data. None if an error occurs.

        Raises:
            Exception: If an error occurs while parsing the file.
        """
        self.logger.info(f"{log_tag('NASDB','P3K')} Parsing as {csv_type}: {file_path}")

        metadata = {}
        header_row_index = -1

        # 1. Find metadata and header row from the initial content
        for i, line in enumerate(header_content):
            try:
                parts = line.strip().split(",")
                if len(parts) > 1:
                    key = parts[0].strip()
                    val = parts[1].strip()
                    if "Record Length" in key:
                        metadata["record_length"] = int(val)
                    elif "Sample Interval" in key:
                        metadata["time_resolution"] = float(val)

                # Find the actual header row
                if "TIME" in line.upper() and "CH" in line.upper():
                    header_row_index = i
            except (ValueError, IndexError):
                # Ignore lines that don't fit the key-value or header pattern
                continue

        if header_row_index == -1:
            self.logger.error(
                f"{log_tag('NASDB','P3K')} Could not find a valid header row ('TIME', 'CH...') in {file_path}"
            )
            return None

        self.logger.info(
            f"{log_tag('NASDB','P3K')} Found header at line {header_row_index + 1}. Metadata: {metadata}"
        )

        # 2. Read the actual data using pandas, skipping to the data section
        read_target = file_path
        temp_file = None
        try:
            if self.access_mode == "remote":
                self.logger.info(
                    f"{log_tag('NASDB','P3K')} Downloading remote file to temp location: {file_path}"
                )
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                self.sftp_client.get(file_path, temp_file.name)
                read_target = temp_file.name
            else:
                # Use pathlib.Path for safe local file handling
                read_target = Path(read_target)

            # Treat the detected header line as the header row (align with MSO58 handling)
            # Try C engine first; fallback to python engine if needed
            try:
                p = ensure_str_path(read_target)
                with open(p, "rb") as fh:
                    df = pd.read_csv(
                        fh,
                        skiprows=header_row_index,
                        header=0,
                        encoding="utf-8",
                        on_bad_lines="warn",
                        low_memory=False,
                        encoding_errors="ignore",
                        memory_map=False,
                    )
            except Exception as e_c:
                self.logger.warning(
                    f"{log_tag('NASDB','P3K')} C engine parse failed for standard CSV: {e_c}. Retrying with python engine."
                )
                p = ensure_str_path(read_target)
                with open(p, "rb") as fh:
                    df = pd.read_csv(
                        fh,
                        skiprows=header_row_index,
                        header=0,
                        encoding="utf-8",
                        on_bad_lines="warn",
                        encoding_errors="ignore",
                        engine="python",
                    )

            # Standardize column names: TIME for first column, CH0, CH1, CH2... for the rest
            standardized_cols = ["TIME"] + [
                f"CH{i}" for i in range(len(df.columns) - 1)
            ]
            df.columns = standardized_cols

            # Add metadata to the DataFrame's attributes for later use
            df.attrs["metadata"] = metadata

            return df
        except Exception as e:
            self.logger.error(
                f"{log_tag('NASDB','P3K')} Pandas failed to parse {file_path} with detected header. Error: {e}"
            )
            return None
        finally:
            if temp_file:
                temp_file.close()
                os.unlink(temp_file.name)

    def _read_fpga_dat(self, file_path: str, **kwargs) -> pd.DataFrame | None:
        self.logger.info(f"{log_tag('NASDB','PFPGA')} Parsing as FPGA .dat: {file_path}")
        read_target = file_path
        temp_file = None
        try:
            if self.access_mode == "remote":
                self.logger.info(
                    f"{log_tag('NASDB','PFPGA')} Downloading remote .dat file to temp location: {file_path}"
                )
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".dat")
                self.sftp_client.get(file_path, temp_file.name)
                read_target = temp_file.name
            else:
                # Use pathlib.Path for safe local file handling
                read_target = Path(read_target)

            # Read data first without column names
            p = ensure_str_path(read_target)
            with open(p, "rb") as fh:
                df = pd.read_csv(fh, sep=r"\s+", header=None)

            # Standardize column names: TIME for first column, CH0, CH1, CH2... for the rest
            standardized_cols = ["TIME"] + [
                f"CH{i}" for i in range(len(df.columns) - 1)
            ]
            df.columns = standardized_cols
            df.attrs["source_file_type"] = "dat"
            df.attrs["source_file_format"] = "FPGA"
            return df
        except Exception as e:
            self.logger.error(f"{log_tag('NASDB','PFPGA')} Failed to parse FPGA file {file_path}: {e}")
            return None
        finally:
            if temp_file:
                temp_file.close()
                os.unlink(temp_file.name)

    def _read_matlab_mat(self, file_path: str, **kwargs) -> pd.DataFrame | None:
        """
        Parses a MATLAB .mat file.

        Args:
            file_path (str): The path to the file.

        Returns:
            pd.DataFrame | None: The parsed data. None if an error occurs.

        Raises:
            Exception: If an error occurs while parsing the file.
        """
        self.logger.info(f"{log_tag('NASDB','PMAT')} Parsing as MATLAB .mat: {file_path}")
        try:
            local_mat_path = file_path
            if self.access_mode == "remote":
                # .mat are binary, need to be downloaded whole first.
                self.logger.warning(
                    f"{log_tag('NASDB','PMAT')} Remote .mat files will be downloaded to a temporary local file."
                )
                local_mat_path = Path(self.dumping_folder) / Path(file_path).name
                self.sftp_client.get(file_path, local_mat_path)

            from scipy.io import loadmat

            mat = loadmat(local_mat_path)

            if self.access_mode == "remote":
                os.remove(local_mat_path)  # Clean up temp file

            # This assumes a simple structure, might need adjustment based on actual .mat files
            # Look for a variable that is a numpy array
            data_key = [
                k
                for k in mat
                if isinstance(mat[k], np.ndarray) and not k.startswith("__")
            ][0]
            data = mat[data_key]

            cols = ["TIME"] + [f"CH{i + 1}" for i in range(data.shape[1] - 1)]
            df = pd.DataFrame(data, columns=cols)
            df.attrs["source_file_type"] = "mat"
            df.attrs["source_file_format"] = "MATLAB"
            return df
        except Exception as e:
            self.logger.error(f"{log_tag('NASDB','PMAT')} Failed to parse MATLAB file {file_path}: {e}")
            return None

    def get_data_top(
        self,
        query: Union[int, str, List[Union[int, str]]],
        data_folders: Union[list, str] = None,
        lines: int = 50,
    ) -> str | None:
        """
        Retrieves the first few lines of the first file found for a shot.
        Useful for inspecting headers.

        Args:
            query (Union[int, str, List[Union[int, str]]]): The query to search for.
            data_folders (Union[list, str]): The folders to search in. Default is None.
            lines (int): The number of lines to read. Default is 50.

        Returns:
            str | None: The top N lines of the file. None if an error occurs.

        Raises:
            ConnectionError: If the connection to the NAS cannot be established.
        """
        if not self._is_connected:
            if not self.connect():
                raise ConnectionError("Failed to establish connection to NAS.")

        if isinstance(data_folders, str):
            data_folders = [data_folders]

        file_paths = self.find_files(query, data_folders)
        if not file_paths:
            self.logger.warning(f"{log_tag('NASDB','QTOP')} No files found for query {query} to get header from.")
            return None

        first_file = file_paths[0]
        self.logger.info(f"{log_tag('NASDB','QTOP')} Getting top {lines} lines from: {first_file}")

        if self.access_mode == "local":
            return self._get_data_top_local(first_file, lines)
        else:  # remote
            return self._get_data_top_remote(first_file, lines)

    def _get_data_top_local(self, file_path: str, lines: int) -> str | None:
        """
        Reads the top N lines of a file on the local machine.

        Args:
            file_path (str): The path to the file.
            lines (int): The number of lines to read.

        Returns:
            str | None: The top N lines of the file. None if an error occurs.

        Raises:
            Exception: If an error occurs while reading the file.
        """
        self.logger.info(f"{log_tag('NASDB','QTOP')} Reading top {lines} lines from local file: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                head = [f.readline() for _ in range(lines)]
            return "".join(line for line in head if line)
        except Exception as e:
            self.logger.error(f"{log_tag('NASDB','QTOP')} Error reading top lines from {file_path}: {e}")
            return None

    def _get_data_top_remote(self, file_path: str, lines: int, use_semaphore: bool = True) -> str | None:
        """
        Reads the top N lines of a file on the remote machine.

        Args:
            file_path (str): The path to the file.
            lines (int): The number of lines to read.
            use_semaphore (bool): Whether to use semaphore for thread safety.
                Set to False if already called within a semaphore-protected block to avoid deadlock.

        Returns:
            str | None: The top N lines of the file. None if an error occurs.

        Raises:
            Exception: If an error occurs while reading the file.
        """
        self.logger.info(f"{log_tag('NASDB','QTOPR')} Reading top {lines} lines from remote file...")

        # Generate unique filename to avoid collisions in parallel execution
        script_filename = _generate_unique_script_name(f"head_{int(time.time())}")
        remote_script_path = os.path.join(
            self.remote_temp_dir, script_filename
        ).replace("\\", "/")
        
        # Use semaphore to limit concurrent SSH command executions
        # This allows up to max_concurrent_ssh_commands commands to run simultaneously
        # while preventing too many concurrent connections that could cause issues
        # However, if already called within a semaphore block, skip to avoid deadlock
        semaphore_context = self.ssh_command_semaphore if use_semaphore else nullcontext()
        with semaphore_context:
            # 1. Write the head script to a remote temp file
            # Ensure the remote temp directory exists before trying to write to it
            # Note: use_semaphore=False because we're already in a semaphore block
            self._ensure_remote_dir_exists(self.remote_temp_dir, use_semaphore=False)

            try:
                with self.sftp_client.open(remote_script_path, "w") as f:
                    f.write(REMOTE_HEAD_SCRIPT)
            except Exception as e:
                self.logger.error(f"{log_tag('NASDB','QTOPR')} Failed to write remote script: {e}")
                return None

            # 2. Execute the script
            cmd = f'python "{remote_script_path}" "{file_path}" "{lines}"'

            self.logger.info(f"{log_tag('NASDB','QTOPR')} Executing remote command...")
        
            stdin, stdout, stderr = self.ssh_client.exec_command(cmd)
            # 3. Get output and check for errors
            output = stdout.read().decode("utf-8", errors="ignore")
            err_output = stderr.read().decode("utf-8", errors="ignore").strip()

            if err_output:
                self.logger.error(f"{log_tag('NASDB','QTOPR')} Remote script error: {err_output}")

            # 4. Cleanup remote script (always execute, regardless of errors)
            try:
                self.sftp_client.remove(remote_script_path)
            except Exception as e:
                self.logger.warning(
                    f"{log_tag('NASDB','QTOPR')} Failed to remove remote script {remote_script_path}: {e}"
                )

            if output:
                return output
            else:
                self.logger.warning(f"{log_tag('NASDB','QTOPR')} No data returned from remote script for get_data_top.")
                return None

    def disconnect(self):
        if self.sftp_client:
            self.sftp_client.close()
            self.sftp_client = None
        if self.ssh_client:
            self.ssh_client.close()
            self.ssh_client = None
        self._is_connected = False
        self.logger.info(f"{log_tag('NASDB','DISC')} Disconnected.")

    def __enter__(self):
        if not self.connect():
            raise ConnectionError("Failed to establish connection to NAS.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def _setup_logger(self):
        """Sets up the logger."""
        self.logger = LogManager().get_logger(
            f"{__name__}.{self.__class__.__name__}", level="INFO"
        )
