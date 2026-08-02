"""
LangGraph wiring for the prototype pipeline:

    script_analyst -> scene_design -> cinematography -> critic_cinematography
        --(any scene fails, retries left)--> cinematography  [loop]
        --(all pass, or out of retries)--> editing -> final_critic -> END

This intentionally matches the "decomposition framework" diagram from the
architecture doc, trimmed to the stages a v1 prototype needs: no
rendering, no sound, no VFX yet. The point of this prototype is to prove
the *reasoning* pipeline works before spending anything on generation.
"""

from langgraph.graph import StateGraph, END

from film_agents.schemas import ProjectState
from film_agents import agents


def node_script_analyst(state: ProjectState) -> dict:
    breakdown = agents.script_analyst_agent(state.script_text)
    return {
        "breakdown": breakdown,
        "log": state.log + [f"[Script Analyst] extracted {len(breakdown.scenes)} scenes"],
    }


def node_scene_design(state: ProjectState) -> dict:
    briefs = [agents.scene_design_agent(scene) for scene in state.breakdown.scenes]
    return {
        "intent_briefs": briefs,
        "log": state.log + [f"[Scene Design] wrote {len(briefs)} intent briefs"],
    }


def node_cinematography(state: ProjectState) -> dict:
    """Generates shot lists. On a revision pass, only regenerates shot
    lists for scenes the critic flagged, reusing feedback as guidance."""
    scene_by_num = {s.scene_number: s for s in state.breakdown.scenes}
    brief_by_num = {b.scene_number: b for b in state.intent_briefs}

    failing_feedback = {
        c.scene_number: c.feedback
        for c in state.cinematography_critiques
        if not c.passes
    }

    existing_by_num = {sl.scene_number: sl for sl in state.shot_lists}
    new_shot_lists = []

    for scene_number, scene in scene_by_num.items():
        brief = brief_by_num[scene_number]
        needs_revision = scene_number in failing_feedback
        if scene_number in existing_by_num and not needs_revision:
            new_shot_lists.append(existing_by_num[scene_number])
            continue
        feedback = failing_feedback.get(scene_number)
        shot_list = agents.cinematography_agent(scene, brief, prior_feedback=feedback)
        new_shot_lists.append(shot_list)

    revision_count = state.cinematography_revision_count + (1 if failing_feedback else 0)
    return {
        "shot_lists": new_shot_lists,
        "cinematography_revision_count": revision_count,
        "log": state.log + [f"[Cinematography] produced/updated {len(new_shot_lists)} shot lists"],
    }


def node_critic_cinematography(state: ProjectState) -> dict:
    brief_by_num = {b.scene_number: b for b in state.intent_briefs}
    reports = []
    for shot_list in state.shot_lists:
        brief = brief_by_num[shot_list.scene_number]
        artifact_desc = "\n".join(
            f"- {s.shot_number} ({s.shot_type}, movement={s.movement}): "
            f"{s.description} [serves: {s.narrative_purpose}]"
            for s in shot_list.shots
        )
        report = agents.critic_agent(brief, "cinematography", artifact_desc)
        reports.append(report)
    return {
        "cinematography_critiques": reports,
        "log": state.log + [f"[Critic] scored {len(reports)} shot lists"],
    }


def route_after_cinematography_critic(state: ProjectState) -> str:
    any_failing = any(not c.passes for c in state.cinematography_critiques)
    if any_failing and state.cinematography_revision_count < state.max_revisions:
        return "revise"
    return "advance"


def node_editing(state: ProjectState) -> dict:
    brief_by_num = {b.scene_number: b for b in state.intent_briefs}
    sequences = []
    for shot_list in state.shot_lists:
        brief = brief_by_num[shot_list.scene_number]
        sequences.append(agents.editing_agent(brief, shot_list))
    return {
        "edited_sequences": sequences,
        "log": state.log + [f"[Editing] assembled {len(sequences)} scene sequences"],
    }


def node_final_critic(state: ProjectState) -> dict:
    brief_by_num = {b.scene_number: b for b in state.intent_briefs}
    reports = []
    for seq in state.edited_sequences:
        brief = brief_by_num[seq.scene_number]
        artifact_desc = (
            f"Shot order: {seq.shot_order}\n"
            f"Pacing notes: {seq.pacing_notes}\n"
            f"Transitions: {seq.transitions_in_out}\n"
        )
        reports.append(agents.critic_agent(brief, "editing", artifact_desc))
    return {
        "final_critiques": reports,
        "log": state.log + [f"[Critic] scored {len(reports)} edited sequences"],
    }


def build_reasoning_graph():
    graph = StateGraph(ProjectState)

    graph.add_node("script_analyst", node_script_analyst)
    graph.add_node("scene_design", node_scene_design)
    graph.add_node("cinematography", node_cinematography)
    graph.add_node("critic_cinematography", node_critic_cinematography)
    graph.add_node("editing", node_editing)
    graph.add_node("final_critic", node_final_critic)

    graph.set_entry_point("script_analyst")
    graph.add_edge("script_analyst", "scene_design")
    graph.add_edge("scene_design", "cinematography")
    graph.add_edge("cinematography", "critic_cinematography")

    graph.add_conditional_edges(
        "critic_cinematography",
        route_after_cinematography_critic,
        {"revise": "cinematography", "advance": "editing"},
    )

    graph.add_edge("editing", "final_critic")
    graph.add_edge("final_critic", END)

    return graph.compile()
