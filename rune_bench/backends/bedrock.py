"""AWS Bedrock LLM backend stub.

Scope:      SaaS API
Docs:       https://docs.aws.amazon.com/bedrock/latest/APIReference/
            https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html
Ecosystem:  AWS

Implementation notes:
- Install:  pip install boto3
- Auth:     AWS credentials via standard boto3 chain:
            - env vars: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN
            - OR IAM role / instance profile
            - region is REQUIRED (no default): BackendCredentials.extra["region"]
- SDK:      import boto3
            client = boto3.client("bedrock-runtime", region_name=credentials.extra["region"])
- Models:   anthropic.claude-3-5-sonnet-*, amazon.titan-*, meta.llama3-*, mistral.*, ...
- Invoke:   client.invoke_model(modelId=model, body=json.dumps({...}))
- Capabilities: client.get_foundation_model(modelIdentifier=model)
            Returns modelDetails.responseStreamingSupported, contextLength (if present)
- BackendCredentials:
            api_key  = None (uses boto3 credential chain)
            base_url = None
            extra    = {"region": "us-east-1"}
"""

from rune_bench.backends.base import BackendCredentials, ModelCapabilities


class BedrockBackend:
    """LLM backend for AWS Bedrock.

    Vendor quirk: region is mandatory and passed via BackendCredentials.extra["region"].
    """

    def __init__(self, credentials: BackendCredentials) -> None:
        if not credentials.extra.get("region"):
            raise ValueError("BedrockBackend requires BackendCredentials.extra['region'] to be set.")
        self._credentials = credentials

    def get_model_capabilities(self, model: str) -> ModelCapabilities:
        """Fetch model metadata from AWS Bedrock."""
        raise NotImplementedError(
            "BedrockBackend is not yet implemented. "
            "See https://docs.aws.amazon.com/bedrock/latest/APIReference/ for details."
        )
