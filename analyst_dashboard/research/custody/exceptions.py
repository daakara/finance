"""
ARX Terminal — Evidence Custody Exception Hierarchy
Authority: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence
"""

class CustodyError(Exception):
    """Base exception for all evidence custody errors."""
    pass

class VaultConfigurationError(CustodyError):
    """Raised when vault root is missing, unset, or configured inside Git repository."""
    pass

class VaultPermissionError(CustodyError):
    """Raised when vault filesystem permissions or ACLs fail security check."""
    pass

class CryptoTamperError(CustodyError):
    """Raised when ciphertext authentication tag fails or data is corrupted."""
    pass

class CustodyKeyError(CustodyError):
    """Raised when encryption key is missing, invalid length, or invalid format."""
    pass

class IntakeValidationError(CustodyError):
    """Raised when submission violates size limits, schema, taxonomy, or trial set."""
    pass

class BlindingError(CustodyError):
    """Raised when unauthorized rater cross-access or premature scoring is attempted."""
    pass

class LedgerIntegrityError(CustodyError):
    """Raised when receipt hash-chain validation fails or tampering is detected."""
    pass

class WithdrawalError(CustodyError):
    """Raised when participant withdrawal or data deletion workflow fails."""
    pass
