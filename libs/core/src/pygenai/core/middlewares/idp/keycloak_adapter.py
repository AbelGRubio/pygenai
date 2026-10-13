"""Keycloak Identity Provider Adapter."""

from typing import Any, ClassVar

import jwt

from observe_core.security.idp.idp_adapter import IDPAdapter


class KeycloakAdapter(IDPAdapter):
    """Identity Provider adapter specifically for Keycloak.

    This adapter handles Keycloak-specific OIDC endpoint construction,
    RS256 algorithm usage, and specialized role extraction from the
    'resource_access' claim.
    """

    # Keycloak typically uses RSA-based signatures
    algorithms: ClassVar = ["RS256"]

    def __init__(self, jwks_url: str, audience: str, issuer: str, verify_token: bool = False) -> None:
        """Initializes the Keycloak adapter by formatting standard OIDC URLs.

        Note:
            In Keycloak, the JWKS and Issuer URLs are relative to the 'realm'.
            The constructor builds these paths using the provided issuer name.

        Args:
            jwks_url: The base URL of the Keycloak server.
            audience: The Client ID (Resource) defined in Keycloak.
            issuer: The name of the Keycloak Realm.
            verify_token: boolean to verify token
        """
        # Formats: {base}/realms/{realm}/protocol/openid-connect/certs
        key_jwks_url = jwks_url + f"/realms/{issuer}/protocol/openid-connect/certs"
        key_audience = audience
        self._audience = audience
        # Formats: {base}/realms/{realm}
        key_issuer = f"{jwks_url}/realms/{issuer}"

        super().__init__(key_jwks_url, key_audience, key_issuer, verify_token)

    def get_roles(self, payload: dict[str, Any]) -> set[str]:
        """Extracts roles from Keycloak's specific 'resource_access' claim.

        In Keycloak, roles are usually nested as follows:
        resource_access -> {client_id} -> roles -> [list of roles]

        Args:
            payload: The decoded JWT payload.

        Returns:
            set[str]: A set of roles assigned to the user for this specific client.
        """
        return set(payload.get("resource_access", {}).get(self.audience, {}).get("roles", []))

    def get_payload_verified(self, token: str) -> dict[str, Any]:
        """Decodes and verifies the token with a fallback for the 'account' audience.

        Keycloak tokens often include 'account' in the audience claim (aud). If
        verification fails with the primary client ID, this method attempts
        verification using 'account' as the audience before falling back to
        the standard IDPAdapter logic.

        Args:
            token: The raw JWT string.

        Returns:
            dict: The verified and decoded payload.
        """
        try:
            key = self._get_key(token)
            return jwt.decode(
                token,
                key,
                algorithms=self.algorithms,
                audience="account",
                issuer=self.issuer,
            )
        except Exception:
            # If 'account' audience verification fails, try the default logic
            return super().get_payload_verified(token)
