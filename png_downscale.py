"""
Pure-Python PNG downscaler — no external dependencies.
Downscales PNG to max_dim pixels on the longest side using simple pixel averaging.
"""
import struct, zlib, io, base64


def downscale_png_to_data_url(path: str, max_dim: int = 512) -> str:
    """Read a PNG, downscale to max_dim, return as base64 data URL."""
    with open(path, "rb") as f:
        raw = f.read()

    # Parse PNG chunks
    sig = raw[:8]
    if sig != b'\x89PNG\r\n\x1a\n':
        # Not a valid PNG — return as-is (might be JPEG or other)
        b64 = base64.b64encode(raw).decode()
        suffix = path.rsplit(".", 1)[-1].lower()
        if suffix == "jpg":
            suffix = "jpeg"
        return f"data:image/{suffix};base64,{b64}"

    # Find IHDR chunk
    pos = 8
    width = height = 0
    bit_depth = color_type = 0
    idat_chunks = []
    palette_data = b""

    while pos < len(raw):
        length = struct.unpack(">I", raw[pos:pos+4])[0]
        chunk_type = raw[pos+4:pos+8]
        chunk_data = raw[pos+8:pos+8+length]

        if chunk_type == b"IHDR":
            width = struct.unpack(">I", chunk_data[0:4])[0]
            height = struct.unpack(">I", chunk_data[4:8])[0]
            bit_depth = chunk_data[8]
            color_type = chunk_data[9]
        elif chunk_type == b"IDAT":
            idat_chunks.append(chunk_data)
        elif chunk_type == b"PLTE":
            palette_data = chunk_data
        elif chunk_type == b"IEND":
            break

        pos += 12 + length

    if width == 0 or not idat_chunks:
        # Can't parse — return raw file as-is
        b64 = base64.b64encode(raw).decode()
        return f"data:image/png;base64,{b64}"

    # If already small enough, return as-is
    if max(width, height) <= max_dim:
        b64 = base64.b64encode(raw).decode()
        return f"data:image/png;base64,{b64}"

    # Decode pixel data
    decompressed = zlib.decompress(b"".join(idat_chunks))

    # Determine bytes per pixel and row stride
    if color_type == 0:  # Grayscale
        channels = 1
        has_alpha = False
    elif color_type == 2:  # RGB
        channels = 3
        has_alpha = False
    elif color_type == 3:  # Indexed
        channels = 1
        has_alpha = False
    elif color_type == 4:  # Grayscale + Alpha
        channels = 2
        has_alpha = True
    elif color_type == 6:  # RGBA
        channels = 4
        has_alpha = True
    else:
        b64 = base64.b64encode(raw).decode()
        return f"data:image/png;base64,{b64}"

    bytes_per_pixel = channels * (bit_depth // 8)
    row_stride = width * bytes_per_pixel + 1  # +1 for filter byte

    # Extract raw pixels (strip filter byte from each row)
    pixels = bytearray()
    for y in range(height):
        row_start = y * row_stride + 1  # skip filter byte
        row_data = decompressed[row_start:row_start + width * bytes_per_pixel]
        if color_type == 3:  # Indexed — need to resolve palette
            for px_idx in range(width):
                idx = row_data[px_idx] if px_idx < len(row_data) else 0
                pal_start = idx * 3
                if pal_start + 2 < len(palette_data):
                    pixels.extend(palette_data[pal_start:pal_start+3])
                else:
                    pixels.extend(b'\x00\x00\x00')
        else:
            pixels.extend(row_data)

    # Calculate new dimensions
    scale = max_dim / max(width, height)
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))

    # Downsample by nearest-neighbor
    out_channels = 3  # Always output RGB
    out_pixels = bytearray(new_w * new_h * out_channels)
    for y in range(new_h):
        src_y = int(y / scale)
        for x in range(new_w):
            src_x = int(x / scale)
            src_offset = (src_y * width + src_x) * out_channels
            dst_offset = (y * new_w + x) * out_channels
            for c in range(out_channels):
                if src_offset + c < len(pixels):
                    out_pixels[dst_offset + c] = pixels[src_offset + c]

    # Encode as new PNG
    def make_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + chunk + crc

    # Build IHDR
    ihdr_data = struct.pack(">IIBBBBB", new_w, new_h, 8, 2, 0, 0, 0)
    ihdr = make_chunk(b"IHDR", ihdr_data)

    # Build IDAT
    raw_rows = bytearray()
    for y in range(new_h):
        raw_rows.append(0)  # filter byte: None
        row_start = y * new_w * out_channels
        raw_rows.extend(out_pixels[row_start:row_start + new_w * out_channels])
    compressed = zlib.compress(bytes(raw_rows))
    idat = make_chunk(b"IDAT", compressed)

    iend = make_chunk(b"IEND", b"")

    out_png = sig + ihdr + idat + iend
    b64 = base64.b64encode(bytes(out_png)).decode()
    return f"data:image/png;base64,{b64}"
