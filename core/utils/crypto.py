import json
import logging
from pathlib import Path

from base58 import b58encode
from nacl.signing import SigningKey


def get_public_key_from_private_bytes(pv_bytes: bytes) -> str:
    pv = SigningKey(pv_bytes)
    pb_bytes = bytes(pv.verify_key)
    return b58encode(pb_bytes).decode()


def save_keypair(pv_bytes: bytes, output_dir: str) -> str:
    pv = SigningKey(pv_bytes)
    pb_bytes = bytes(pv.verify_key)
    pubkey = b58encode(pb_bytes).decode()
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    file_path = Path(output_dir) / f"{pubkey}.json"
    with open(file_path, 'wb') as f:
        f.write(json.dumps(list(pv_bytes + pb_bytes)).encode())
            
    logging.info(f"Found: {pubkey}")
    return pubkey
