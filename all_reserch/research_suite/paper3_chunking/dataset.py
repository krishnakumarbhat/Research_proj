from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkingDocument:
    doc_id: str
    title: str
    sentences: tuple[str, ...]
    global_context: str

    @property
    def text(self) -> str:
        return " ".join(self.sentences)


@dataclass(frozen=True)
class ChunkingQuery:
    query_id: str
    text: str
    doc_id: str
    support_sentences: tuple[int, ...]
    query_type: str


SCIENTISTS = [
    "Elena Brooks",
    "Marcus Lee",
    "Priya Raman",
    "Jonah Hart",
    "Sofia Perez",
    "Ada Winters",
    "Theo Chen",
    "Mila Novak",
    "Ava Patel",
    "Rina Das",
    "Noah Flynn",
    "Luca Meyer",
]
PROJECTS = [
    "Atlas Rover",
    "Nimbus Drone",
    "Vector Beacon",
    "Pulse Grid",
    "Helio Scope",
    "Axiom Forge",
    "Kestrel Relay",
    "Quanta Surveyor",
    "Cinder Pilot",
    "Blue Orbit Node",
    "Signal Ridge Cart",
    "Harbor Memory Rover",
]
LABS = [
    "Nova Lab",
    "Helix Institute",
    "Arc Signal Lab",
    "Verdant Dynamics",
    "Northline Lab",
    "Harbor Memory Lab",
]
BATTERIES = [
    "lithium-silicate battery",
    "graphene pack",
    "solid-state battery",
    "hybrid supercapacitor",
    "ceramic fuel core",
    "compressed hydrogen cell",
]
SENSORS = [
    "lidar guidance stack",
    "spectral camera rig",
    "optical relay array",
    "adaptive planner",
    "thermal sensor ring",
    "carbon sampler",
]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
SITES = ["Aurora Bay", "Meridian Port", "Cobalt Yard", "North Shelf", "Ridge Point", "Delta Hangar"]
REVIEWERS = [
    "Iris Cole",
    "Jon Park",
    "Maya Singh",
    "Owen Lake",
    "Nina Frost",
    "Leo Grant",
]
ARCHIVE_ROOMS = ["Room A3", "Vault B1", "Cabinet C7", "Shelf D4", "Archive E2", "Locker F5"]


def build_dataset(quick: bool = False) -> tuple[list[ChunkingDocument], list[ChunkingQuery]]:
    document_count = 6 if quick else 12
    documents: list[ChunkingDocument] = []
    queries: list[ChunkingQuery] = []

    for index in range(document_count):
        scientist = SCIENTISTS[index % len(SCIENTISTS)]
        project = PROJECTS[index % len(PROJECTS)]
        partner_project = PROJECTS[(index + 3) % len(PROJECTS)]
        lab = LABS[index % len(LABS)]
        partner_lab = LABS[(index + 2) % len(LABS)]
        partner_lead = SCIENTISTS[(index + 4) % len(SCIENTISTS)]
        battery = BATTERIES[index % len(BATTERIES)]
        sensor = SENSORS[index % len(SENSORS)]
        day = DAYS[index % len(DAYS)]
        site = SITES[index % len(SITES)]
        reviewer = REVIEWERS[index % len(REVIEWERS)]
        archive_room = ARCHIVE_ROOMS[index % len(ARCHIVE_ROOMS)]

        sentences = (
            f"{scientist} founded {project} at {lab}.",
            f"She said the first public launch would happen on {day}.",
            f"The vehicle uses a {battery} and a {sensor}.",
            f"Because it shares the same controller as {partner_project}, the review board lowered the risk score.",
            f"{partner_lead} leads {partner_project} for {partner_lab}.",
            f"The board later approved the test after {scientist} submitted a revised checklist.",
            f"The agreement names {site} as the backup site.",
            f"That site also stores the calibration kit used by the rover.",
            f"A quarterly memo notes that {reviewer} reviews the climate package.",
            f"He archived the last audit in {archive_room}.",
        )
        global_context = (
            f"{project} summary: founder {scientist}; lab {lab}; partner {partner_project}; "
            f"backup site {site}; climate reviewer {reviewer}."
        )
        doc_id = f"doc_{index:02d}"
        documents.append(
            ChunkingDocument(
                doc_id=doc_id,
                title=project,
                sentences=sentences,
                global_context=global_context,
            )
        )
        queries.extend(
            [
                ChunkingQuery(
                    query_id=f"{doc_id}_q0",
                    text=f"Who said the first public launch of {project} would happen on {day}?",
                    doc_id=doc_id,
                    support_sentences=(0, 1),
                    query_type="pronoun_launch",
                ),
                ChunkingQuery(
                    query_id=f"{doc_id}_q1",
                    text=f"What battery does {project} use?",
                    doc_id=doc_id,
                    support_sentences=(2,),
                    query_type="single_fact",
                ),
                ChunkingQuery(
                    query_id=f"{doc_id}_q2",
                    text=f"Which partner project shares the same controller as {project}?",
                    doc_id=doc_id,
                    support_sentences=(3,),
                    query_type="single_fact",
                ),
                ChunkingQuery(
                    query_id=f"{doc_id}_q3",
                    text=f"Who submitted the revised checklist for {project}?",
                    doc_id=doc_id,
                    support_sentences=(0, 5),
                    query_type="coreference_long",
                ),
                ChunkingQuery(
                    query_id=f"{doc_id}_q4",
                    text=f"Where is the backup site for {project}?",
                    doc_id=doc_id,
                    support_sentences=(6,),
                    query_type="single_fact",
                ),
                ChunkingQuery(
                    query_id=f"{doc_id}_q5",
                    text=f"What does the backup site store for {project}?",
                    doc_id=doc_id,
                    support_sentences=(6, 7),
                    query_type="adjacent_dependency",
                ),
                ChunkingQuery(
                    query_id=f"{doc_id}_q6",
                    text=f"Who archived the last audit for {project}?",
                    doc_id=doc_id,
                    support_sentences=(8, 9),
                    query_type="pronoun_audit",
                ),
            ]
        )
    return documents, queries
