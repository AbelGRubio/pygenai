"""Identity Provider Base Adapter Module."""

from abc import ABC
from collections.abc import Callable
from typing import Any, ClassVar

import jwt
from jwt import PyJWKClient


class IDPAdapter(ABC):
    """Abstract Base Class for Identity Provider (IDP) adapters.

    This class manages the logic for fetching public keys from a JWKS endpoint,
    verifying JWT signatures, and extracting user authorization roles.

    Attributes:
        jwks_url (str): URL to the Identity Provider's JWKS endpoint.
        audience (str): The expected 'aud' claim in the JWT.
        issuer (str): The expected 'iss' claim from the IDP.
        algorithms (list): Supported signing algorithms (defaults to ES384).
    """

    get_payload: Callable[[str], dict[str, Any]]
    algorithms: ClassVar = ["ES384"]

    def __init__(self, jwks_url: str, audience: str, issuer: str, verify_token: bool = False) -> None:
        """Initializes the adapter and determines the verification strategy.

        The verification strategy is determined by the 'VERIFY_TOKEN' environment
        variable. If 'true', signatures are cryptographically validated.
        """
        self.jwks_url = jwks_url
        self.audience = audience
        self.issuer = issuer
        self.jwks_client = PyJWKClient(jwks_url)
        self.verify_token = verify_token

        # Strategy Pattern: Switch between verified and unverified decoding
        # verify = os.environ.get("VERIFY_TOKEN", "false").lower() == "true"
        if self.verify_token:
            self.get_payload = self.get_payload_verified
        else:
            self.get_payload = self.get_payload_unverified

    @staticmethod
    def get_payload_unverified(token: str) -> dict[str, Any]:
        """Decodes the token without signature or claim verification.

        Warning: Use only for local development or debugging.

        Args:
            token (str): The decoded JWT claims.

        Returns:
             dict[str, Any]: Decoded token.
        """
        return jwt.decode(token, options={"verify_signature": False})

    def get_payload_verified(self, token: str) -> dict[str, Any]:
        """Decodes and validates the token using the IDP's public keys.

        Performs checks for:
        1. Signature validity (using remote JWKS).
        2. Audience ('aud') match.
        3. Issuer ('iss') match.

        Args:
            token (str): The decoded JWT claims.

        Returns:
             dict[str, Any]: Decoded token.
        """
        key = self._get_key(token)
        return jwt.decode(
            token,
            key,
            algorithms=self.algorithms,
            audience=self.audience,
            issuer=self.issuer,
        )

    def get_roles(self, payload: dict[str, Any]) -> set[str]:
        """Extracts authorization roles from the decoded JWT payload.

        Args:
            payload (dict): The decoded JWT claims.

        Returns:
            set[str]: A set of roles for efficient permission checking.
        """
        return set(payload.get("roles", []))

    def _get_key(self, token: str) -> Any:
        """Fetches the specific signing key from the JWKS endpoint based on the token's 'kid'."""
        return self.jwks_client.get_signing_key_from_jwt(token).key
