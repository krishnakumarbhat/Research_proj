from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RagDocument:
    doc_id: str
    text: str
    doc_type: str
    topic: str
    linked_ids: tuple[str, ...]


@dataclass(frozen=True)
class RagQuery:
    query_id: str
    text: str
    relevant_ids: tuple[str, ...]
    family_id: str
    query_type: str


TOPICS = ["robotics", "vision", "energy", "climate", "biotech", "logistics"]
PROJECT_PREFIXES = ["Atlas", "Nimbus", "Vector", "Pulse", "Helio", "Axiom", "Kestrel", "Quanta"]
PROJECT_SUFFIXES = ["Rover", "Grid", "Beacon", "Pilot", "Scope", "Forge"]
LAB_NAMES = [
    "Nova Lab",
    "Helix Institute",
    "Cinder Works",
    "Blue Orbit Lab",
    "Tidal Systems Lab",
    "Arc Signal Lab",
    "Northline Lab",
    "Verdant Dynamics",
    "Harbor Memory Lab",
    "Signal Ridge Lab",
    "Copper Field Lab",
    "Lumen Pattern Lab",
]
COMPONENTS = [
    "lidar guidance stack",
    "optical relay array",
    "adaptive planner",
    "thermo sensor ring",
    "edge inference board",
    "carbon sampler",
    "spectral camera rig",
    "routing mesh controller",
]
ENERGY_CELLS = [
    "lithium-silicate cell",
    "graphene pack",
    "solid-state battery",
    "hybrid supercapacitor",
    "ceramic fuel core",
    "compressed hydrogen cell",
]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
SITES = ["Aurora Bay", "Meridian Port", "Cobalt Yard", "North Shelf", "Ridge Point", "Delta Hangar"]
DIRECTORS = [
    "Elena Brooks",
    "Marcus Lee",
    "Priya Raman",
    "Jonah Hart",
    "Sofia Perez",
    "Ada Winters",
    "Noah Flynn",
    "Mila Novak",
    "Rina Das",
    "Theo Chen",
    "Luca Meyer",
    "Ava Patel",
]
PROTOCOLS = [
    "Blue Shield",
    "Quiet Grid",
    "Vector Gate",
    "Clean Room Eight",
    "Signal Lock",
    "Drift Check",
]


def _project_names(limit: int) -> list[str]:
    names = [f"{prefix} {suffix}" for prefix in PROJECT_PREFIXES for suffix in PROJECT_SUFFIXES]
    return names[:limit]


def build_dataset(quick: bool = False, seed: int = 42) -> tuple[list[RagDocument], list[RagQuery]]:
    rng = np.random.default_rng(seed)
    project_count = 18 if quick else 36
    project_names = _project_names(project_count)
    documents: list[RagDocument] = []
    project_records: list[dict[str, str]] = []
    lab_projects: dict[str, list[str]] = {lab_name: [] for lab_name in LAB_NAMES}

    for index, project_name in enumerate(project_names):
        topic = TOPICS[index % len(TOPICS)]
        lab_name = LAB_NAMES[index % len(LAB_NAMES)]
        component = COMPONENTS[index % len(COMPONENTS)]
        energy_cell = ENERGY_CELLS[(index * 2) % len(ENERGY_CELLS)]
        launch_day = DAYS[index % len(DAYS)]
        site = SITES[(index * 3) % len(SITES)]
        partner_name = project_names[(index + 5) % len(project_names)]
        project_id = f"project_{index:02d}"
        text = (
            f"{project_name} is a {topic} program managed by {lab_name}. "
            f"It uses a {component} and a {energy_cell}. "
            f"Its launch window is {launch_day} at {site}. "
            f"The platform shares navigation telemetry with {partner_name}."
        )
        documents.append(
            RagDocument(
                doc_id=project_id,
                text=text,
                doc_type="project",
                topic=topic,
                linked_ids=(),
            )
        )
        project_records.append(
            {
                "project_id": project_id,
                "project_name": project_name,
                "lab_name": lab_name,
                "topic": topic,
                "component": component,
                "energy_cell": energy_cell,
                "launch_day": launch_day,
                "site": site,
                "partner_name": partner_name,
            }
        )
        lab_projects[lab_name].append(project_name)

    lab_index_map: dict[str, str] = {}
    for index, lab_name in enumerate(LAB_NAMES):
        topic = TOPICS[index % len(TOPICS)]
        director = DIRECTORS[index % len(DIRECTORS)]
        protocol = PROTOCOLS[index % len(PROTOCOLS)]
        site = SITES[(index + 2) % len(SITES)]
        supervised = lab_projects[lab_name][:3] if lab_projects[lab_name] else ["no active projects"]
        lab_id = f"lab_{index:02d}"
        lab_index_map[lab_name] = lab_id
        text = (
            f"{lab_name} is directed by {director}. "
            f"The lab focuses on {topic} systems and oversees {', '.join(supervised)}. "
            f"Its safety protocol is {protocol}, and the backup facility is {site}."
        )
        documents.append(
            RagDocument(
                doc_id=lab_id,
                text=text,
                doc_type="lab",
                topic=topic,
                linked_ids=tuple(),
            )
        )

    document_index = {document.doc_id: document for document in documents}
    refreshed_documents: list[RagDocument] = []
    for record in project_records:
        project_id = record["project_id"]
        lab_id = lab_index_map[record["lab_name"]]
        partner_id = next(
            item["project_id"] for item in project_records if item["project_name"] == record["partner_name"]
        )
        project_document = document_index[project_id]
        refreshed_documents.append(
            RagDocument(
                doc_id=project_document.doc_id,
                text=project_document.text,
                doc_type=project_document.doc_type,
                topic=project_document.topic,
                linked_ids=(lab_id, partner_id),
            )
        )
    refreshed_documents.extend(document for document in documents if document.doc_type == "lab")
    documents = refreshed_documents

    intents: list[dict[str, object]] = []
    for rank, record in enumerate(project_records, start=1):
        hot_weight = 1.0 / rank
        project_id = str(record["project_id"])
        lab_id = lab_index_map[str(record["lab_name"])]
        partner_id = next(
            item["project_id"] for item in project_records if item["project_name"] == record["partner_name"]
        )
        intents.extend(
            [
                {
                    "family_id": f"{project_id}_component",
                    "query_type": "project_fact",
                    "weight": hot_weight,
                    "relevant_ids": (project_id,),
                    "templates": [
                        "Which component does {project_name} use?",
                        "What hardware stack powers {project_name}?",
                        "Name the core component inside {project_name}.",
                    ],
                    "fields": record,
                },
                {
                    "family_id": f"{project_id}_launch",
                    "query_type": "project_fact",
                    "weight": hot_weight * 0.85,
                    "relevant_ids": (project_id,),
                    "templates": [
                        "When is the launch window for {project_name}?",
                        "Which day does {project_name} launch?",
                        "Tell me the launch day for {project_name}.",
                    ],
                    "fields": record,
                },
                {
                    "family_id": f"{project_id}_lab_protocol",
                    "query_type": "linked_pair",
                    "weight": hot_weight * 0.72,
                    "relevant_ids": (project_id, lab_id),
                    "templates": [
                        "Which lab manages {project_name} and what protocol does it enforce?",
                        "For {project_name}, name the overseeing lab and its safety protocol.",
                        "What lab owns {project_name}, and which protocol belongs to that lab?",
                    ],
                    "fields": record,
                },
                {
                    "family_id": f"{project_id}_partner",
                    "query_type": "linked_pair",
                    "weight": hot_weight * 0.64,
                    "relevant_ids": (project_id, partner_id),
                    "templates": [
                        "Which partner shares navigation telemetry with {project_name}?",
                        "Name the project linked to {project_name} through shared telemetry.",
                        "What partner project is paired with {project_name} for navigation telemetry?",
                    ],
                    "fields": record,
                },
            ]
        )

    for index, lab_name in enumerate(LAB_NAMES):
        director = DIRECTORS[index % len(DIRECTORS)]
        protocol = PROTOCOLS[index % len(PROTOCOLS)]
        lab_id = lab_index_map[lab_name]
        lab_fields = {
            "lab_name": lab_name,
            "director": director,
            "protocol": protocol,
        }
        intents.extend(
            [
                {
                    "family_id": f"{lab_id}_director",
                    "query_type": "lab_fact",
                    "weight": 0.35,
                    "relevant_ids": (lab_id,),
                    "templates": [
                        "Who directs {lab_name}?",
                        "Name the director of {lab_name}.",
                        "Which person leads {lab_name}?",
                    ],
                    "fields": lab_fields,
                },
                {
                    "family_id": f"{lab_id}_protocol",
                    "query_type": "lab_fact",
                    "weight": 0.30,
                    "relevant_ids": (lab_id,),
                    "templates": [
                        "What protocol belongs to {lab_name}?",
                        "Which safety protocol does {lab_name} enforce?",
                        "Tell me the protocol used by {lab_name}.",
                    ],
                    "fields": lab_fields,
                },
            ]
        )

    weights = np.array([float(intent["weight"]) for intent in intents], dtype=float)
    weights /= weights.sum()
    query_count = 180 if quick else 540
    queries: list[RagQuery] = []
    for query_index in range(query_count):
        intent = intents[int(rng.choice(len(intents), p=weights))]
        template = rng.choice(intent["templates"])
        text = str(template).format(**intent["fields"])
        queries.append(
            RagQuery(
                query_id=f"query_{query_index:04d}",
                text=text,
                relevant_ids=tuple(intent["relevant_ids"]),
                family_id=str(intent["family_id"]),
                query_type=str(intent["query_type"]),
            )
        )
    return documents, queries
