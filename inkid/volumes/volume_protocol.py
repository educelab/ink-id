"""Module for storing the Volume Protocol networking specification and classes."""
import socket
import struct
import dataclasses
import typing

MAGIC = 0xf6fcdac0
VOLPKG_SZ = 64
VOLUME_SZ = 64
V1 = 1


@dataclasses.dataclass
class RequestHdr:
    """Class for storing a request header for a request to a VC Volume Server."""
    num_requests: int
    magic: int = MAGIC
    version: int = V1

    @staticmethod
    def struct_format():
        """Return the format to convert this object into a C-style struct."""
        # The 0I at the end is to align the overall struct to 4-bytes.
        return "IBB0I"

    def to_struct(self):
        """Convert this object into a C-style struct."""
        return struct.pack(self.struct_format(), self.magic, self.version, self.num_requests)


@dataclasses.dataclass
class RequestArgs:
    """Class for storing a single request to a VC Volume Server."""
    # pylint: disable=too-many-instance-attributes
    volpkg: str
    volume: str
    center_x: float
    center_y: float
    center_z: float
    sampling_r_x: float
    sampling_r_y: float
    sampling_r_z: float
    axis_x_x: float = 1.0
    axis_x_y: float = 0.0
    axis_x_z: float = 0.0
    axis_y_x: float = 0.0
    axis_y_y: float = 1.0
    axis_y_z: float = 0.0
    axis_z_x: float = 0.0
    axis_z_y: float = 0.0
    axis_z_z: float = 1.0
    sampling_interval: float = 1.0

    @staticmethod
    def struct_format():
        """Return the format to convert this object into a C-style struct."""
        return f"{VOLPKG_SZ}s{VOLUME_SZ}s16f"

    def to_struct(self):
        """Convert this object into a C-style struct."""
        return struct.pack(self.struct_format(), str.encode(self.volpkg), str.encode(self.volume),
                           self.center_x, self.center_y, self.center_z,
                           self.axis_x_x, self.axis_x_y, self.axis_x_z,
                           self.axis_y_x, self.axis_y_y, self.axis_y_z,
                           self.axis_z_x, self.axis_z_y, self.axis_z_z,
                           self.sampling_r_x, self.sampling_r_y, self.sampling_r_z,
                           self.sampling_interval)


@dataclasses.dataclass
class ResponseArgs:
    """Class for storing a response from a VC Volume Server."""
    volpkg: str
    volume: str
    extent_x: int
    extent_y: int
    extent_z: int
    size: int

    @staticmethod
    def struct_format():
        """Return the format to convert this object into a C-style struct."""
        return f"{VOLPKG_SZ}s{VOLUME_SZ}s4I"

    @classmethod
    def from_struct(cls, struct_data):
        """Create an object from a C-style struct."""
        (volpkg, volume, extent_x, extent_y, extent_z, size) = struct.unpack(
            cls.struct_format(), struct_data)
        return cls(volpkg=volpkg.rstrip(b'\0').decode(),
                   volume=volume.rstrip(b'\0').decode(), extent_x=extent_x,
                   extent_y=extent_y, extent_z=extent_z, size=size)


def get_subvolumes(requests: typing.List[RequestArgs],
                   server: typing.Tuple[str, int] = ("127.0.0.1", 8087)):
    """Returns an array of one or more requested subvolumes from a volume server."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(server)
    request_hdr = RequestHdr(len(requests))
    sock.send(request_hdr.to_struct())
    for request in requests:
        sock.send(request.to_struct())
    responses = []
    for _ in requests:
        response_args = ResponseArgs.from_struct(
            sock.recv(struct.calcsize(ResponseArgs.struct_format())))
        response_data = bytearray()
        while len(response_data) < response_args.size:
            tmp_data = sock.recv(4096)
            response_data.extend(tmp_data)
        responses.append((response_args, response_data))
    sock.close()
    return responses
