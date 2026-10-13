"""Identity Provider Factory Module."""

import os

from observe_core.logger import get_logger
from observe_core.security.idp.idp_adapter import IDPAdapter
from observe_core.security.idp.keycloak_adapter import KeycloakAdapter

logger = get_logger(__name__)


class IDPFactory:
    """Factory class to instantiate the appropriate Identity Provider adapter.

    This class centralizes the creation logic for IDP adapters, enabling
    plug-and-play support for different authentication services based on
    system configuration.
    """

    def __init__(self, provider: str = "logto", verify_token: bool = False) -> None:
        """Init instance."""
        self.provider = provider.lower()
        self.verify_token = verify_token

    def get_idp(self) -> IDPAdapter:
        """Retrieves and initializes an IDP adapter instance based on environment variables.

        The method reads the `IDP_PROVIDER` variable to select the implementation
        class and passes the necessary OIDC parameters (URL, Audience, Issuer)
        to the constructor.

        Returns:
            IDPAdapter: An initialized instance of a concrete IDP adapter.

        Raises:
            ValueError: If the `IDP_PROVIDER` specified is not supported.

        Environment Variables Used:
            IDP_PROVIDER: Type of provider (e.g., 'keycloak', 'logto', 'authentik', 'cognito').
            IDP_URL: The JWKS endpoint for token validation.
            IDP_AUDIENCE: The expected audience claim in the JWT.
            IDP_ISSUER: The expected issuer claim in the JWT.
        """
        # Determine the provider from environment, defaulting to 'logto'
        # provider = os.getenv("IDP_PROVIDER", "logto").lower()

        # Mapping of provider keys to their respective Adapter classes
        idps_ = {
            "keycloak": KeycloakAdapter,
        }

        idp_adapter_class = idps_.get(self.provider)

        if not idp_adapter_class:
            logger.error(f"Unknown IDP provider: '{self.provider}'. Supported providers are: {list(idps_.keys())}")
            raise ValueError(f"Unknown IDP provider: '{self.provider}'. Supported providers are: {list(idps_.keys())}")

        # Initialize the selected adapter with OIDC configuration
        return idp_adapter_class(
            jwks_url=os.getenv("IDP_URL", "http://localhost:3001/oidc/jwks"),
            audience=os.getenv("IDP_AUDIENCE", "mcp-client"),
            issuer=os.getenv("IDP_ISSUER", "http://localhost:3001/oidc"),
            verify_token=self.verify_token,
        )
