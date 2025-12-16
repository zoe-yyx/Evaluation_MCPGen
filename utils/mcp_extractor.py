"""
MCP Server 工具提取模块
从 MCP Server 的 server.py 中提取工具信息

迁移自: tasks/workflow_shuffling_mcp.py
"""

import sys
import inspect
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class MCPToolInfo:
    """MCP工具信息数据类"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema
        }


class MCPServerToolsExtractor:
    """
    通过MCP协议从server.py提取工具信息
    支持两种方式：
    1. 使用fastmcp Client连接运行中的server (推荐)
    2. 直接导入server模块获取工具定义
    """
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.server_path = project_path / "mcp_server" / "server.py"
        self.tools: List[MCPToolInfo] = []
    
    async def extract_tools_via_mcp_client(self, timeout: float = 30.0) -> List[MCPToolInfo]:
        """
        方法1：使用 uv run 启动 MCP Server 获取工具信息
        这会自动处理 pyproject.toml 中的依赖，实现环境隔离
        """
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            import os
            
            env = os.environ.copy()
            env["UV_NO_PROGRESS"] = "1" 
            env["PYTHONUNBUFFERED"] = "1"
            
            # 配置启动参数：使用 uv run 执行 server.py
            # 注意：cwd 设置为 project_path，这样 uv 才能找到 pyproject.toml
            server_params = StdioServerParameters(
                command="uv",
                args=["run", str(self.server_path)],
                cwd=str(self.project_path),
                env=env
            )
            
            extracted_tools = []
            
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # 初始化 MCP 协议
                    await session.initialize()
                    tools_response = await session.list_tools()
                    
                    for tool in tools_response.tools:
                        tool_info = MCPToolInfo(
                            name=tool.name,
                            description=tool.description or "",
                            input_schema=tool.inputSchema or {}
                        )
                        extracted_tools.append(tool_info)
            
            self.tools = extracted_tools
            return extracted_tools
            
        except FileNotFoundError:
             print("⚠️ 'uv' command not found. Please ensure uv is installed and in your PATH.")
             return []
        except Exception as e:
            print(f"⚠️ MCP Client (uv run) connection failed: {e}")
            if "exit code" in str(e):
                print(f"   💡 Tip: Check if {self.project_path}/pyproject.toml exists and is valid.")
                print(f"   💡 Tip: Try running 'uv run {self.server_path}' manually in that directory to debug.")
            return []

    async def extract_tools_via_import(self) -> List[MCPToolInfo]:
        """
        方法2：直接导入server模块获取工具定义
        """
        import importlib.util
        
        try:
            spec = importlib.util.spec_from_file_location("server", self.server_path)
            server_module = importlib.util.module_from_spec(spec)
            
            original_path = sys.path.copy()
            sys.path.insert(0, str(self.project_path))
            sys.path.insert(0, str(self.project_path / "mcp_server"))
            
            try:
                spec.loader.exec_module(server_module)
            except ImportError as e:
                print(f"   ❌ Import failed due to missing dependency: {e}")
                print(f"   👉 Please run: uv add {e.name}")
                raise e
            finally:
                sys.path = original_path
            
            tools = []
            
            # 查找fastmcp的mcp实例
            mcp_instance = None
            for name, obj in vars(server_module).items():
                # fastmcp 实例通常包含 _tools 属性
                if hasattr(obj, '_tools'): 
                    mcp_instance = obj
                    break
            
            if mcp_instance:
                # 从fastmcp实例获取工具
                if hasattr(mcp_instance, '_tools'):
                    for tool_name, tool_func in mcp_instance._tools.items():
                        description = tool_func.__doc__ or ""
                        input_schema = self._extract_function_schema(tool_func)
                        tools.append(MCPToolInfo(
                            name=tool_name,
                            description=description.strip(),
                            input_schema=input_schema
                        ))
            
            self.tools = tools
            return tools
            
        except Exception as e:
            print(f"❌ Failed to import server module: {e}")
            return []

    def _extract_function_schema(self, func) -> Dict:
        """从函数签名提取参数schema"""
        schema = {"type": "object", "properties": {}, "required": []}
        try:
            sig = inspect.signature(func)
            hints = func.__annotations__ if hasattr(func, '__annotations__') else {}
            for param_name, param in sig.parameters.items():
                if param_name in ('self', 'cls', 'ctx'): # fastmcp 可能会注入 ctx
                    continue
                prop = {"type": "string"}
                if param_name in hints:
                    hint = hints[param_name]
                    if hint == int: prop["type"] = "integer"
                    elif hint == float: prop["type"] = "number"
                    elif hint == bool: prop["type"] = "boolean"
                    elif hint == list: prop["type"] = "array"
                    elif hint == dict: prop["type"] = "object"
                schema["properties"][param_name] = prop
                if param.default == inspect.Parameter.empty:
                    schema["required"].append(param_name)
        except Exception:
            pass
        return schema

    async def extract_tools(self, method: str = "auto") -> List[MCPToolInfo]:
        """
        提取工具信息的统一入口
        
        Args:
            method: 提取方法 - "auto", "mcp_client", "import"
        
        Returns:
            List[MCPToolInfo]: 提取的工具列表
        """
        if not self.server_path.exists():
            print(f"❌ Server file not found: {self.server_path}")
            return []
        
        # 优先使用 fastmcp client，其次尝试 import
        methods = {
            "mcp_client": self.extract_tools_via_mcp_client,
            "import": self.extract_tools_via_import
        }
        
        if method == "auto":
            for method_name, extractor in methods.items():
                print(f"   Trying {method_name} method...")
                tools = await extractor()
                if tools:
                    print(f"   ✅ Successfully extracted {len(tools)} tools via {method_name}")
                    return tools
            print("   ❌ All extraction methods failed")
            return []
        else:
            if method in methods:
                return await methods[method]()
            else:
                raise ValueError(f"Unknown extraction method: {method}")
