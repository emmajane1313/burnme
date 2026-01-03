import asyncio
import json
from pathlib import Path
from typing import Optional
import httpx
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.backends import default_backend
import os
import base64
from datetime import datetime
from pydantic import BaseModel

APP_SECRET = b'burnmewhileimhot_y2k_secret_2026_sparkles'

class MP4PMetadata(BaseModel):
    id: str
    expiresAt: int
    salt: str
    iv: str
    authTag: str
    createdAt: int
    burned: bool
    burnedAt: Optional[int] = None
    daydreamApiKey: Optional[str] = None
    synthedSalt: Optional[str] = None
    synthedIv: Optional[str] = None
    synthedAuthTag: Optional[str] = None
    promptsUsed: Optional[list] = None

class MP4PData(BaseModel):
    metadata: MP4PMetadata
    encryptedVideo: str
    encryptedSynthedVideo: Optional[str] = None
    signature: str

def derive_key(salt: str) -> bytes:
    material = APP_SECRET + salt.encode()
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt.encode(),
        iterations=100000,
        backend=default_backend()
    )
    return kdf.derive(material)

def create_signature(metadata: dict) -> str:
    h = hmac.HMAC(APP_SECRET, hashes.SHA256(), backend=default_backend())
    h.update(json.dumps(metadata).encode())
    return h.finalize().hex()

def verify_signature(metadata: dict, signature: str) -> bool:
    expected = create_signature(metadata)
    return expected == signature

async def encrypt_video(video_data: bytes, expires_at: int, video_id: str) -> MP4PData:
    salt = os.urandom(16).hex()
    iv = os.urandom(16)
    key = derive_key(salt)

    cipher = Cipher(
        algorithms.AES(key),
        modes.GCM(iv),
        backend=default_backend()
    )
    encryptor = cipher.encryptor()

    encrypted_video = encryptor.update(video_data) + encryptor.finalize()
    auth_tag = encryptor.tag

    metadata = MP4PMetadata(
        id=video_id,
        expiresAt=expires_at,
        salt=salt,
        iv=iv.hex(),
        authTag=auth_tag.hex(),
        createdAt=int(datetime.now().timestamp() * 1000),
        burned=False
    )

    metadata_dict = metadata.model_dump()
    signature = create_signature(metadata_dict)

    return MP4PData(
        metadata=metadata,
        encryptedVideo=base64.b64encode(encrypted_video).decode(),
        signature=signature
    )

async def decrypt_video(mp4p_data: MP4PData) -> bytes:
    metadata_dict = mp4p_data.metadata.model_dump()

    if not verify_signature(metadata_dict, mp4p_data.signature):
        raise ValueError("Invalid signature - file may be corrupted or tampered")

    key = derive_key(mp4p_data.metadata.salt)
    iv = bytes.fromhex(mp4p_data.metadata.iv)
    auth_tag = bytes.fromhex(mp4p_data.metadata.authTag)
    encrypted_buffer = base64.b64decode(mp4p_data.encryptedVideo)

    cipher = Cipher(
        algorithms.AES(key),
        modes.GCM(iv, auth_tag),
        backend=default_backend()
    )
    decryptor = cipher.decryptor()

    decrypted_video = decryptor.update(encrypted_buffer) + decryptor.finalize()
    return decrypted_video

async def add_synthed_video(
    mp4p_data: MP4PData,
    synthed_video_data: bytes,
    prompts_used: list
) -> MP4PData:
    salt = os.urandom(16).hex()
    iv = os.urandom(16)
    key = derive_key(salt)

    cipher = Cipher(
        algorithms.AES(key),
        modes.GCM(iv),
        backend=default_backend()
    )
    encryptor = cipher.encryptor()

    encrypted_synthed = encryptor.update(synthed_video_data) + encryptor.finalize()
    auth_tag = encryptor.tag

    mp4p_data.metadata.synthedSalt = salt
    mp4p_data.metadata.synthedIv = iv.hex()
    mp4p_data.metadata.synthedAuthTag = auth_tag.hex()
    mp4p_data.metadata.promptsUsed = prompts_used
    mp4p_data.encryptedSynthedVideo = base64.b64encode(encrypted_synthed).decode()

    metadata_dict = mp4p_data.metadata.model_dump()
    mp4p_data.signature = create_signature(metadata_dict)

    return mp4p_data

async def decrypt_synthed_video(mp4p_data: MP4PData) -> Optional[bytes]:
    if not mp4p_data.encryptedSynthedVideo:
        return None

    metadata_dict = mp4p_data.metadata.model_dump()
    if not verify_signature(metadata_dict, mp4p_data.signature):
        raise ValueError("Invalid signature - file may be corrupted or tampered")

    key = derive_key(mp4p_data.metadata.synthedSalt)
    iv = bytes.fromhex(mp4p_data.metadata.synthedIv)
    auth_tag = bytes.fromhex(mp4p_data.metadata.synthedAuthTag)
    encrypted_buffer = base64.b64decode(mp4p_data.encryptedSynthedVideo)

    cipher = Cipher(
        algorithms.AES(key),
        modes.GCM(iv, auth_tag),
        backend=default_backend()
    )
    decryptor = cipher.decryptor()

    decrypted_synthed = decryptor.update(encrypted_buffer) + decryptor.finalize()
    return decrypted_synthed

def should_show_synthed(mp4p_data: MP4PData) -> bool:
    now = int(datetime.now().timestamp() * 1000)
    return now >= mp4p_data.metadata.expiresAt and mp4p_data.encryptedSynthedVideo is not None

async def burn_video(
    mp4p_data: MP4PData,
    api_key: str,
    video_buffer: bytes
) -> MP4PData:
    async with httpx.AsyncClient() as client:
        files = {'video': ('video.mp4', video_buffer, 'video/mp4')}
        headers = {'Authorization': f'Bearer {api_key}'}

        response = await client.post(
            'https://api.daydream.com/anonymize',
            files=files,
            headers=headers,
            timeout=60.0
        )
        response.raise_for_status()

        anonymized_video = base64.b64decode(response.json()['video'])

    salt = os.urandom(16).hex()
    iv = os.urandom(16)
    key = derive_key(salt)

    cipher = Cipher(
        algorithms.AES(key),
        modes.GCM(iv),
        backend=default_backend()
    )
    encryptor = cipher.encryptor()
    encrypted_video = encryptor.update(anonymized_video) + encryptor.finalize()
    auth_tag = encryptor.tag

    mp4p_data.metadata.burned = True
    mp4p_data.metadata.burnedAt = int(datetime.now().timestamp() * 1000)
    mp4p_data.metadata.salt = salt
    mp4p_data.metadata.iv = iv.hex()
    mp4p_data.metadata.authTag = auth_tag.hex()
    mp4p_data.encryptedVideo = base64.b64encode(encrypted_video).decode()

    metadata_dict = mp4p_data.metadata.model_dump()
    mp4p_data.signature = create_signature(metadata_dict)

    return mp4p_data