import sys
import types


def _install_cryptography_shim():
    if "cryptography" in sys.modules:
        return

    crypto = types.ModuleType("cryptography")
    exceptions = types.ModuleType("cryptography.exceptions")
    hazmat = types.ModuleType("cryptography.hazmat")
    primitives = types.ModuleType("cryptography.hazmat.primitives")
    asymmetric = types.ModuleType("cryptography.hazmat.primitives.asymmetric")
    ed25519 = types.ModuleType("cryptography.hazmat.primitives.asymmetric.ed25519")
    kdf = types.ModuleType("cryptography.hazmat.primitives.kdf")
    hkdf = types.ModuleType("cryptography.hazmat.primitives.kdf.hkdf")
    ciphers = types.ModuleType("cryptography.hazmat.primitives.ciphers")
    aead = types.ModuleType("cryptography.hazmat.primitives.ciphers.aead")
    hashes = types.ModuleType("cryptography.hazmat.primitives.hashes")

    class InvalidSignature(Exception):
        pass

    class Ed25519PublicKey:
        @classmethod
        def from_public_bytes(cls, data):
            return cls()

        def verify(self, signature, data):
            raise InvalidSignature("cryptography shim cannot verify signatures")

    class HKDF:
        def __init__(self, *args, **kwargs):
            pass

        def derive(self, key_material):
            raise RuntimeError("cryptography shim cannot derive keys")

    class AESGCM:
        def __init__(self, *args, **kwargs):
            pass

        def encrypt(self, *args, **kwargs):
            raise RuntimeError("cryptography shim cannot encrypt")

        def decrypt(self, *args, **kwargs):
            raise RuntimeError("cryptography shim cannot decrypt")

    class SHA256:
        pass

    exceptions.InvalidSignature = InvalidSignature
    ed25519.Ed25519PublicKey = Ed25519PublicKey
    hkdf.HKDF = HKDF
    aead.AESGCM = AESGCM
    hashes.SHA256 = SHA256

    sys.modules.update(
        {
            "cryptography": crypto,
            "cryptography.exceptions": exceptions,
            "cryptography.hazmat": hazmat,
            "cryptography.hazmat.primitives": primitives,
            "cryptography.hazmat.primitives.asymmetric": asymmetric,
            "cryptography.hazmat.primitives.asymmetric.ed25519": ed25519,
            "cryptography.hazmat.primitives.kdf": kdf,
            "cryptography.hazmat.primitives.kdf.hkdf": hkdf,
            "cryptography.hazmat.primitives.ciphers": ciphers,
            "cryptography.hazmat.primitives.ciphers.aead": aead,
            "cryptography.hazmat.primitives.hashes": hashes,
        }
    )


_install_cryptography_shim()
