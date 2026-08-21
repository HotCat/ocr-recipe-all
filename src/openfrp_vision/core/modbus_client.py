from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any


@dataclass(frozen=True)
class ModbusConfig:
    protocol: str = "rtu"
    serial_port: str = "/dev/ttyUSB0"
    baudrate: int = 9600
    parity: str = "N"
    bytesize: int = 8
    stopbits: int = 1
    host: str = "192.168.0.68"
    tcp_port: int = 4998
    slave: int = 1
    timeout_s: float = 0.15

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "ModbusConfig":
        timeout_ms = float(params.get("modbus_timeout_ms", 150) or 150)
        return cls(
            protocol=str(params.get("modbus_protocol", "rtu")).lower(),
            serial_port=str(params.get("modbus_serial_port", "/dev/ttyUSB0")),
            baudrate=int(params.get("modbus_baudrate", 9600) or 9600),
            parity=str(params.get("modbus_parity", "N") or "N").upper()[:1],
            bytesize=int(params.get("modbus_bytesize", 8) or 8),
            stopbits=int(params.get("modbus_stopbits", 1) or 1),
            host=str(params.get("modbus_host", "192.168.0.68")),
            tcp_port=int(params.get("modbus_tcp_port", 4998) or 4998),
            slave=int(params.get("modbus_slave", 1) or 1),
            timeout_s=max(0.01, timeout_ms / 1000.0),
        )


class ModbusClientError(RuntimeError):
    pass


class _ClientBox:
    def __init__(self, config: ModbusConfig) -> None:
        self.config = config
        self.client = self._create_client(config)
        self.lock = threading.Lock()

    def _create_client(self, config: ModbusConfig) -> Any:
        try:
            from pymodbus.client import ModbusSerialClient, ModbusTcpClient
        except ImportError:
            try:
                from pymodbus.client.sync import ModbusSerialClient, ModbusTcpClient
            except ImportError as exc:
                raise ModbusClientError("pymodbus is not installed; install openfrp-vision[modbus] or pymodbus") from exc

        if config.protocol == "tcp":
            return ModbusTcpClient(config.host, port=config.tcp_port, timeout=config.timeout_s)
        if config.protocol != "rtu":
            raise ModbusClientError(f"unsupported Modbus protocol: {config.protocol}")
        return ModbusSerialClient(
            port=config.serial_port,
            baudrate=config.baudrate,
            parity=config.parity,
            bytesize=config.bytesize,
            stopbits=config.stopbits,
            timeout=config.timeout_s,
        )

    def connect(self) -> None:
        connected = getattr(self.client, "connected", False)
        if connected:
            return
        ok = self.client.connect()
        if ok is False:
            raise ModbusClientError(f"could not connect Modbus {self.config.protocol}")

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    def read_register(self, address: int) -> int:
        with self.lock:
            self.connect()
            method = self.client.read_holding_registers
            try:
                result = method(int(address), 1, slave=self.config.slave)
            except TypeError:
                result = method(int(address), 1, unit=self.config.slave)
            self._raise_if_error(result, f"read register {address}")
            registers = getattr(result, "registers", None) or []
            if not registers:
                raise ModbusClientError(f"read register {address} returned no data")
            return int(registers[0])

    def write_register(self, address: int, value: int) -> None:
        with self.lock:
            self.connect()
            method = self.client.write_register
            try:
                result = method(int(address), int(value), slave=self.config.slave)
            except TypeError:
                result = method(int(address), int(value), unit=self.config.slave)
            self._raise_if_error(result, f"write register {address}")

    def _raise_if_error(self, result: Any, action: str) -> None:
        if result is None:
            raise ModbusClientError(f"Modbus {action} returned no response")
        is_error = getattr(result, "isError", None)
        if callable(is_error) and is_error():
            raise ModbusClientError(f"Modbus {action} failed: {result}")


_CLIENTS: dict[ModbusConfig, _ClientBox] = {}
_LOCK = threading.Lock()


def _client(config: ModbusConfig) -> _ClientBox:
    with _LOCK:
        box = _CLIENTS.get(config)
        if box is None:
            box = _ClientBox(config)
            _CLIENTS[config] = box
        return box


def read_register(config: ModbusConfig, address: int) -> int:
    return _client(config).read_register(address)


def write_register(config: ModbusConfig, address: int, value: int) -> None:
    _client(config).write_register(address, value)


def read_trigger(params: dict[str, Any]) -> bool:
    config = ModbusConfig.from_params(params)
    address = int(params.get("modbus_trigger_register", 100) or 100)
    value = int(params.get("modbus_trigger_value", 999) or 999)
    return read_register(config, address) == value


def write_decision(params: dict[str, Any], passed: bool) -> tuple[int, int]:
    config = ModbusConfig.from_params(params)
    address_key = "modbus_ok_register" if passed else "modbus_ng_register"
    default_address = 110 if passed else 120
    address = int(params.get(address_key, default_address) or default_address)
    value = int(params.get("modbus_write_value", 999) or 999)
    write_register(config, address, value)
    return address, value


def close_all() -> None:
    with _LOCK:
        boxes = list(_CLIENTS.values())
        _CLIENTS.clear()
    for box in boxes:
        box.close()
