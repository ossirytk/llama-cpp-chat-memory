import asyncio
import signal

from conversation_manager import ConveresationManager

conversation_manager = ConveresationManager()


class GracefulExit(SystemExit):
    code = 1


def raise_graceful_exit(*args):
    loop.stop()
    print("Chat closed")
    raise GracefulExit()


async def main() -> None:
    character_name = conversation_manager.get_character_name()
    while True:
        query = input("User: ")
        print(f"{character_name}: ", end="")
        await conversation_manager.ask_question_test(query)
        print("\n")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    signal.signal(signal.SIGINT, raise_graceful_exit)
    signal.signal(signal.SIGTERM, raise_graceful_exit)
    background_tasks = set()
    task = loop.create_task(main())
    background_tasks.add(task)
    try:
        # asyncio.run(main())
        loop.run_until_complete(task)
    except GracefulExit:
        pass
    finally:
        loop.close()
