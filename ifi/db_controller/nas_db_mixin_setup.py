#!/usr/bin/env python3
"""
NasDB Setup and Connection Mixin
================================

This mixin is responsible for setting up the NAS connection and configuring the NAS paths.

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

import configparser
import os
import threading
import time
from contextlib import nullcontext
from pathlib import Path

import paramiko

from ..utils.log_manager import log_tag
from .nas_db_base import NasDBBase


class NasDBMixinSetup(NasDBBase):
    """Configuration, connection lifecycle, and logger setup."""

    def __init__(self, config_path: str = "ifi/config.ini"):
        self._setup_logger(component=__name__)

        if not Path(config_path).exists():
            self.logger.error(f"{log_tag('NASDB','CONFIG')} Config file not found: '{config_path}'")
            raise FileNotFoundError(
                f"{log_tag('NASDB','CONFIG')} Config file not found: '{config_path}'"
            )

        config = configparser.ConfigParser()
        config.read(config_path)

        nas_cfg = config["NAS"]
        self.nas_path = nas_cfg.get("path")
        self.nas_mount = nas_cfg.get("mount_point")
        self.nas_user = nas_cfg.get("user")
        self.nas_password = nas_cfg.get("password")

        folders_str = nas_cfg.get("data_folders", "")
        self.default_data_folders = [f.strip() for f in folders_str.split(",") if f.strip()]

        ssh_cfg = config["SSH_NAS"]
        self.ssh_host = ssh_cfg.get("ssh_host")
        self.ssh_port = ssh_cfg.getint("ssh_port")
        self.ssh_user = ssh_cfg.get("ssh_user")
        self.ssh_pkey = Path(ssh_cfg.get("ssh_pkey_path")).expanduser()
        self.remote_temp_dir = ssh_cfg.get("remote_temp_dir", None)

        conn_cfg = config["CONNECTION_SETTINGS"]
        self.ssh_max_retries = conn_cfg.getint("ssh_max_retries", 3)
        self.ssh_connect_timeout = conn_cfg.getfloat("ssh_connect_timeout", 10.0)
        self.max_concurrent_ssh_commands = conn_cfg.getint("max_concurrent_ssh_commands", 2)

        if config.has_section("LOCAL_CACHE"):
            cache_cfg = config["LOCAL_CACHE"]
            self.dumping_folder = cache_cfg.get("dumping_folder", "/ifi/results")
            self.max_load_size_gb = cache_cfg.getfloat("max_load_size_gb", 2.0)
        else:
            self.dumping_folder = "/ifi/results"
            self.max_load_size_gb = 2.0

        self.ssh_client = None
        self.sftp_client = None
        self.access_mode = None
        self._file_cache = {}

        self._is_connected = False
        self.ssh_lock = threading.Lock()
        self.ssh_command_semaphore = threading.Semaphore(self.max_concurrent_ssh_commands)
        self.ssh_operation_lock = (
            threading.Lock() if self.max_concurrent_ssh_commands == 1 else None
        )

    def _ensure_remote_dir_exists(self, remote_path: str, use_semaphore: bool = True):
        """Ensure remote directory exists via SFTP."""
        if self.access_mode != "remote" or not self.sftp_client:
            return

        if not remote_path:
            semaphore_context = self.ssh_command_semaphore if use_semaphore else nullcontext()
            with semaphore_context:
                try:
                    home_dir = self.sftp_client.normalize(".")
                    remote_path = os.path.join(home_dir, ".ifi_temp").replace("\\", "/")
                    self.remote_temp_dir = remote_path
                    self.logger.info(
                        f"{log_tag('NASDB','CHKD')} Remote temp directory not configured, "
                        f"using dynamic path: {remote_path}"
                    )
                except Exception as e:
                    self.logger.error(
                        f"{log_tag('NASDB','CHKD')} Could not determine remote home directory: {e}"
                    )
                    raise

        semaphore_context = self.ssh_command_semaphore if use_semaphore else nullcontext()
        with semaphore_context:
            try:
                self.sftp_client.stat(remote_path)
            except FileNotFoundError:
                self.logger.info(
                    f"{log_tag('NASDB','CHKD')} Remote directory '{remote_path}' not found. Creating it."
                )
                self.sftp_client.mkdir(remote_path)
            except Exception as e:
                self.logger.error(
                    f"{log_tag('NASDB','CHKD')} Could not check or create remote directory '{remote_path}': {e}"
                )
                raise

    def connect(self):
        """Establish connection (local mount first, otherwise SSH)."""
        if self._is_connected:
            return True

        with self.ssh_lock:
            if self._is_connected:
                return True

            if Path(self.nas_mount).is_dir():
                self.logger.info(
                    f"{log_tag('NASDB','CONN')} Local NAS mount found at '{self.nas_mount}'. Using direct access."
                )
                self.access_mode = "local"
                self._is_connected = True
                return True

            self.logger.info(
                f"{log_tag('NASDB','CONN')} Local NAS mount not found. Attempting SSH connection."
            )
            self.access_mode = "remote"

            for attempt in range(self.ssh_max_retries):
                try:
                    self.logger.info(
                        f"{log_tag('NASDB','CONN')} SSH connection attempt {attempt + 1}/{self.ssh_max_retries}..."
                    )
                    self.ssh_client = paramiko.SSHClient()
                    self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    self.ssh_client.connect(
                        hostname=self.ssh_host,
                        port=self.ssh_port,
                        username=self.ssh_user,
                        key_filename=str(self.ssh_pkey),
                        timeout=self.ssh_connect_timeout,
                    )
                    self.sftp_client = self.ssh_client.open_sftp()
                    self.logger.info(
                        f"{log_tag('NASDB','CONN')} SSH connection to {self.ssh_host} successful."
                    )

                    self._authenticate_nas_remote()
                    self._is_connected = True
                    return True
                except Exception as e:
                    self.logger.error(
                        f"{log_tag('NASDB','CONN')} SSH connection attempt {attempt + 1} failed: {e}"
                    )
                    self.disconnect()
                    if attempt < self.ssh_max_retries - 1:
                        self.logger.info(f"{log_tag('NASDB','CONN')} Retrying in 3 seconds...")
                        time.sleep(3)

            self._is_connected = False
            self.logger.error(f"{log_tag('NASDB','CONN')} All SSH connection attempts failed.")
            return False

    def _authenticate_nas_remote(self):
        """Authenticate NAS access on remote host with `net use`."""
        auth_cmd = (
            f'net use "{str(self.nas_path)}" /user:{self.nas_user} '
            f"{self.nas_password} /persistent:no"
        )
        self.logger.info(f"{log_tag('NASDB','AUTH')} Authenticating to NAS on remote machine.")
        stdin, stdout, stderr = self.ssh_client.exec_command(auth_cmd)
        _ = stdin

        exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            err = stderr.read().decode("utf-8", errors="ignore").strip()
            if "The command completed successfully" not in err:
                self.logger.warning(
                    f"{log_tag('NASDB','AUTH')} NAS authentication may have failed with exit code {exit_status}: {err}"
                )
            else:
                self.logger.info(f"{log_tag('NASDB','AUTH')} NAS authentication successful.")
        else:
            self.logger.info(f"{log_tag('NASDB','AUTH')} NAS authentication successful.")

    def disconnect(self):
        """Close all active connection handles."""
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
            self.logger.warning(
                f"{log_tag('NASDB','CONN')} Entered NasDB context in offline mode. "
                "Remote NAS features are unavailable, but local results/cache paths can still be used."
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _ = (exc_type, exc_val, exc_tb)
        self.disconnect()
