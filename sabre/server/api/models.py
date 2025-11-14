"""
Pydantic models for SABRE API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ============================================================================
# Connector Management Models
# ============================================================================

class ConnectorCreateRequest(BaseModel):
    """Request model for creating a new connector"""
    name: str = Field(..., description="Unique name for the connector")
    type: str = Field(..., description="Transport type: 'stdio' or 'sse'")
    command: Optional[str] = Field(None, description="Command for stdio transport")
    args: list[str] = Field(default_factory=list, description="Command arguments")
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables")
    url: Optional[str] = Field(None, description="URL for SSE transport")
    headers: dict[str, str] = Field(default_factory=dict, description="HTTP headers for SSE")
    enabled: bool = Field(True, description="Whether connector is enabled")
    timeout: int = Field(30, description="Operation timeout in seconds")


class ConnectorUpdateRequest(BaseModel):
    """Request model for updating a connector (all fields optional)"""
    name: Optional[str] = None
    command: Optional[str] = None
    args: Optional[list[str]] = None
    env: Optional[dict[str, str]] = None
    url: Optional[str] = None
    headers: Optional[dict[str, str]] = None
    enabled: Optional[bool] = None
    timeout: Optional[int] = None


class ConnectorResponse(BaseModel):
    """Response model for connector info"""
    id: str
    name: str
    type: str
    enabled: bool
    connected: bool
    tools_count: int
    created_at: str
    updated_at: str
    source: str  # "yaml" or "api"


class ConnectorDetailResponse(ConnectorResponse):
    """Detailed connector response with configuration"""
    command: Optional[str] = None
    args: list[str] = []
    url: Optional[str] = None
    timeout: int = 30


class ToolResponse(BaseModel):
    """Response model for tool info"""
    name: str
    description: str
    signature: str
    server_name: str
    input_schema: dict


class ConnectorToolsResponse(BaseModel):
    """Response model for connector tools"""
    connector_id: str
    connector_name: str
    tools: list[ToolResponse]
