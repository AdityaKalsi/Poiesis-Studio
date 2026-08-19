import sqlite3
import time

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.sqlite import SqliteSaver


# ---------------------------------------------------------
# STATE
# ---------------------------------------------------------

class State(TypedDict):
    completed: Annotated[list[str], lambda a, b: a + b]


# ---------------------------------------------------------
# ROUTER
# ---------------------------------------------------------

def route_tasks(state: State):

    tasks = ["task_1", "task_2", "task_3"]

    return [
        Send(
            "run_task",
            {
                "task": task
            }
        )
        for task in tasks
        if task not in state.get("completed", [])
    ]


# ---------------------------------------------------------
# TASK NODE
# ---------------------------------------------------------

def run_task(state):

    task = state["task"]

    print(f"\nStarting {task}")

    # Simulate expensive work
    time.sleep(2)

    # Intentionally crash Task 3
    if task == "task_3":
        print("TASK 3 Completed")
        

    print(f"✅ {task} completed")

    return {
        "completed": [task]
    }


# ---------------------------------------------------------
# GRAPH
# ---------------------------------------------------------

builder = StateGraph(State)

builder.add_node("run_task", run_task)

builder.add_conditional_edges(
    START,
    route_tasks,
    ["run_task"]
)

# IMPORTANT:
# Don't connect run_task -> END here.
#
# The Send tasks are independent executions.
# The checkpoint stores their state updates.


conn = sqlite3.connect(
    "test_checkpoint.db",
    check_same_thread=False
)

checkpointer = SqliteSaver(conn)

graph = builder.compile(
    checkpointer=checkpointer
)


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

thread_id = "send_test_001"

config = {
    "configurable": {
        "thread_id": thread_id
    }
}


print("\n==============================")
print("FIRST RUN")
print("==============================\n")




# ---------------------------------------------------------
# INSPECT CHECKPOINT
# ---------------------------------------------------------

state = graph.get_state(config)


print("\n==============================")
print("CHECKPOINT")
print("==============================")

print(state.values)
print(list(graph.get_state_history(config)))

print("Next:", state.next)


# ---------------------------------------------------------
# RESUME
# ---------------------------------------------------------

print("\n==============================")
print("RESUMING")
print("==============================\n")

try:

    graph.invoke(
        None,
        config=config
    )

except Exception as e:

    print("\nGRAPH FAILED AGAIN:")
    print(e)