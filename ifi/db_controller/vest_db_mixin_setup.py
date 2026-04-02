#!/usr/bin/env python3
"""
VestDB Mixins for setup
=======================

Split mixins used by `VestDB` to keep controller files compact.

Author: J. Wang
Date: 2025-01-16
"""

from __future__ import annotations

import configparser
import time
from pathlib import Path

import pymysql
from sshtunnel import BaseSSHTunnelForwarderError, SSHTunnelForwarder

from ..utils.log_manager import log_tag
from ..utils.vest_utils import format_vest_field_label, load_vest_field_maps
from .vest_db_base import VestDBBase


class VestDBMixinConfig(VestDBBase):
    """Configuration and field-label loading helpers."""

    def _load_field_labels(self, csv_path: Path | None) -> None:
        self._ensure_logger(component=__name__)
        self.field_labels: dict[int, str] = {}
        if csv_path is None or not csv_path.exists():
            self.logger.warning(
                f"{log_tag('VESTD', 'CONFIG')} VEST field label file not specified or not found. "
                "Column names will be field IDs."
            )
            return
        
        # by_id: {field_id: {"field_name": ..., "field_unit": ...}}
        by_id, _ = load_vest_field_maps(csv_path)
        # field_labels: {field_id: "Variable Name [Unit]"}
        self.field_labels = {
            field_id: format_vest_field_label(meta.get("field_name", ""), meta.get("field_unit", ""))
            for field_id, meta in by_id.items()
        }
        self.logger.info(
            f"{log_tag('VESTD', 'CONFIG')} Successfully loaded {len(self.field_labels)} "
            f"VEST field labels from {csv_path}."
        )

    def _load_config(self, config_path: str) -> None:
        self._ensure_logger(component=__name__)
        config_file = Path(config_path)
        if not config_file.exists():
            self.logger.error(
                f"{log_tag('VESTD', 'CONFG')} Configuration file not found at '{config_path}'."
            )
            raise FileNotFoundError(
                f"Configuration file not found at '{config_path}'.\n"
                "Please create it from 'config.ini.template'."
            )

        config = configparser.ConfigParser()
        config.read(config_path)

        db_cfg = config["VEST_DB"]
        self.db_host = db_cfg.get("host")
        self.db_user = db_cfg.get("user")
        self.db_password = db_cfg.get("password")
        self.db_name = db_cfg.get("database")
        self.db_port = db_cfg.getint("port", 3306)
        self.db_connection_args = {
            "host": self.db_host,
            "user": self.db_user,
            "password": self.db_password,
            "database": self.db_name,
            "port": self.db_port,
        }

        self.field_label_file = db_cfg.get("field_label_file", fallback=None)
        label_path: Path | None = None
        if self.field_label_file:
            label_path = Path(self.field_label_file).expanduser()
            if not label_path.is_absolute():
                label_path = (config_file.parent / label_path).resolve()
        self._load_field_labels(label_path)

        self.tunnel_enabled = config.getboolean("SSH_TUNNEL", "enabled", fallback=False)
        self.ssh_config = {}
        self.ssh_max_retries = 1
        if self.tunnel_enabled:
            ssh_cfg = config["SSH_TUNNEL"]
            conn_cfg = config["CONNECTION_SETTINGS"]

            ssh_key_path = Path(ssh_cfg.get("ssh_pkey_path")).expanduser()
            self.logger.info(
                f"{log_tag('VESTD', 'CONFIG')} SSH key path resolved to: {ssh_key_path}"
            )
            if not ssh_key_path.exists():
                self.logger.warning(
                    f"{log_tag('VESTD', 'CONFIG')} SSH private key file does not exist at '{ssh_key_path}'!"
                )

            self.ssh_config = {
                "ssh_address_or_host": (
                    ssh_cfg.get("ssh_host"),
                    ssh_cfg.getint("ssh_port"),
                ),
                "ssh_username": ssh_cfg.get("ssh_user"),
                "ssh_pkey": str(ssh_key_path),
                "remote_bind_address": (ssh_cfg.get("remote_mysql_host"), self.db_port),
                "set_keepalive": 60.0,
            }
            SSHTunnelForwarder.SSH_TIMEOUT = conn_cfg.getfloat("ssh_connect_timeout")
            self.ssh_max_retries = conn_cfg.getint("ssh_max_retries")

        self.direct_connect_timeout = config.getint(
            "CONNECTION_SETTINGS", "direct_connect_timeout", fallback=3
        )
        self.connection = None
        self.tunnel = None


class VestDBMixinConnection(VestDBBase):
    """Connection lifecycle and generic query helpers."""

    def connect(self) -> bool:
        """Connect to VEST DB directly first, then fallback to SSH tunnel."""
        self._ensure_logger(component=__name__)
        if self.connection and self.connection.open:
            return True

        self.logger.info(
            f"{log_tag('VESTD', 'CONN')} Attempting direct connection to VEST DB..."
        )
        try:
            self.connection = pymysql.connect(
                **self.db_connection_args, connect_timeout=self.direct_connect_timeout
            )
            if self.connection.open:
                self.logger.info(
                    f"{log_tag('VESTD', 'CONN')} Direct connection successful."
                )
                return True
        except pymysql.Error as err:
            self.logger.warning(
                f"{log_tag('VESTD', 'CONN')} Direct connection failed: {err}"
            )
            if not self.tunnel_enabled:
                self.logger.error(
                    f"{log_tag('VESTD', 'CONN')} SSH tunnel is disabled. Cannot proceed."
                )
                return False
            self.logger.info(
                f"{log_tag('VESTD', 'CONN')} Direct connection failed. Now attempting fallback to SSH tunnel."
            )

        self.logger.info(
            f"{log_tag('VESTD', 'CONN')} Falling back to SSH tunnel connection..."
        )
        for attempt in range(self.ssh_max_retries):
            try:
                self.logger.info(
                    f"{log_tag('VESTD', 'CONN')} Attempt {attempt + 1}/{self.ssh_max_retries}..."
                )
                self.tunnel = SSHTunnelForwarder(**self.ssh_config)
                self.tunnel.start()
                self.logger.info(
                    f"{log_tag('VESTD', 'CONN')} SSH tunnel established (localhost:{self.tunnel.local_bind_port})."
                )

                tunneled_cfg = self.db_connection_args.copy()
                tunneled_cfg["host"] = "127.0.0.1"
                tunneled_cfg["port"] = self.tunnel.local_bind_port
                self.connection = pymysql.connect(**tunneled_cfg)

                if self.connection.open:
                    self.logger.info(
                        f"{log_tag('VESTD', 'CONN')} MySQL connection through tunnel successful."
                    )
                    return True

            except BaseSSHTunnelForwarderError as e:
                self.logger.error(
                    f"{log_tag('VESTD', 'CONN')} SSH Tunnel Error on attempt {attempt + 1}: {e}",
                    exc_info=True,
                )
                self.disconnect()
            except pymysql.Error as e:
                self.logger.error(
                    f"{log_tag('VESTD', 'CONN')} MySQL Connection Error (via Tunnel) on attempt {attempt + 1}: {e}",
                    exc_info=True,
                )
                self.disconnect()
            except Exception as e:
                self.logger.error(
                    f"{log_tag('VESTD', 'CONN')} Unexpected error during "
                    f"SSH tunnel connection on attempt {attempt + 1}: {e}",
                    exc_info=True,
                )
                self.disconnect()

            if attempt < self.ssh_max_retries - 1:
                self.logger.info(f"{log_tag('VESTD', 'CONN')} Retrying in 3 seconds...")
                time.sleep(3)

        self.logger.error(
            f"{log_tag('VESTD', 'CONN')} Failed to establish a database connection."
        )
        return False

    def disconnect(self) -> None:
        """Close DB and tunnel connections."""
        self._ensure_logger(component=__name__)
        if self.connection and self.connection.open:
            self.connection.close()
            self.logger.info(f"{log_tag('VESTD', 'DISC')} MySQL connection closed.")
        self.connection = None

        if self.tunnel and self.tunnel.is_active:
            self.tunnel.stop()
            self.logger.info(f"{log_tag('VESTD', 'DISC')} SSH tunnel closed.")
        self.tunnel = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()

    def query(self, sql_query: str, params=None):
        """Execute SQL query and return all rows."""
        self._ensure_logger(component=__name__)
        if not (self.connection and self.connection.open):
            self.logger.error(f"{log_tag('VESTD', 'QRY')} Not connected to the database.")
            return None

        try:
            with self.connection.cursor() as cursor:
                if params is not None:
                    cursor.execute(sql_query, params)
                else:
                    cursor.execute(sql_query)
                results = cursor.fetchall()
                self.logger.info(
                    f"{log_tag('VESTD', 'QRY')} Query executed successfully. Returned {len(results)} rows."
                )
                return results
        except pymysql.Error as e:
            self.logger.error(f"{log_tag('VESTD', 'QRY')} Database query error: {e}")
            return None
        except Exception as e:
            self.logger.error(
                f"{log_tag('VESTD', 'QRY')} Unexpected error during query execution: {e}"
            )
            return None
