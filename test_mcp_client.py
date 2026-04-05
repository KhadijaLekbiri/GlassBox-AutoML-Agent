import asyncio
import json

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client


async def main():
    # 👇 FIX: pass command as a LIST (no keyword)
    async with stdio_client(
        ["python", "-m", "agent.tool_schema"]
    ) as (read, write):

        async with ClientSession(read, write) as session:
            await session.initialize()

            print("\n📦 Available tools:")
            tools = await session.list_tools()
            print(tools)

            print("\n🚀 Calling AutoFit...\n")

            result = await session.call_tool(
                "AutoFit",
                {
                    "csv_path": "titanic_dataset.csv",
                    "target_col": "Survived"
                }
            )

            print("\n✅ RESULT:\n")
            print(result)


asyncio.run(main())