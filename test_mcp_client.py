import asyncio
import json
import sys
import os

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters


async def main():
    # Resolve paths automatically — no hardcoded paths needed
    project_dir   = os.path.dirname(os.path.abspath(__file__))
    python_exe    = sys.executable  # same Python that's running this script
    server_script = os.path.join(project_dir, "run_server.py")

    server_params = StdioServerParameters(
        command=python_exe,
        args=[server_script],
        env={
            "PYTHONUNBUFFERED": "1",
            "GLASSBOX_PROJECT_PATH": project_dir,
        }
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("\n📦 Available tools:")
            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")

            print("\n🚀 Calling AutoFit...\n")

            result = await session.call_tool(
                "AutoFit",
                {
                    "csv_path": "titanic_dataset.csv",
                    "target_col": "Survived",
                    "task_type": "classification",
                    "time_budget": 60,
                    "cv_folds": 5
                }
            )

            print("\n✅ RESULT:\n")
            for block in result.content:
                if hasattr(block, "text"):
                    try:
                        parsed = json.loads(block.text)
                        print(json.dumps(parsed, indent=2))
                    except json.JSONDecodeError:
                        # Non-JSON text block — print as-is
                        print(block.text)


asyncio.run(main())