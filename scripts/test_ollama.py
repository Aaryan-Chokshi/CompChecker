"""Send one live request through the application's Ollama adapter."""

from pydantic import BaseModel, ConfigDict

from backend.core.config import Settings
from backend.integrations.llm.ollama import OllamaLLM


class ConnectionReply(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reply: str


def main() -> None:
    llm = OllamaLLM(Settings())
    try:
        response = llm.generate_structured(
            "Answer accurately in one short sentence.",
            "What is 2 + 2?",
            ConnectionReply,
        )
        print(response.reply)
    finally:
        llm.close()


if __name__ == "__main__":
    main()
