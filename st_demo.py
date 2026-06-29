from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import streamlit as st
import streamlit.components.v1 as components

from embedding import Embeddings
from kd_tree import KDTreeIndex, KDNode
from models import Record
from utils import normalize


class _EChartsNode(TypedDict):
    name: str
    children: list


# Why: visualization nodes need a compact and readable multiline label.
def build_node_label(name: str, size: int, radius: float) -> str:
    """Return a formatted multiline label for an ECharts tree node."""
    return f"{name}\nsize={size}\nradius={radius:.3f}"


def st_echarts(options, height=400, width=600):
    """Функция для отображения графиков ECharts через HTML"""
    if isinstance(height, (int, float)):
        height = f"{int(height)}px"
    if isinstance(width, (int, float)):
        width = f"{int(width)}px"

    html = f"""
        <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
        <div id="echart" style="width:{width}; height:{height};"></div>
        <script>
            var chart = echarts.init(document.getElementById('echart'));
            chart.setOption({json.dumps(options)});
        </script>
        """

    h = int(''.join(c for c in str(height) if c.isdigit()))
    components.html(html, height=h + 20)


# Why: ECharts expects a nested dict structure that must be derived recursively.
def build_tree(
    records: list[Record],
    embedder: Embeddings
):

    tree = KDTreeIndex(dim=384)

    for record in records:
        vector = normalize(embedder.embed(record.text))

        tree.insert(
            vector=vector,
            item_id=str(record.id),
            payload=record
        )

    tree.build()

    return tree


def node_to_echarts(
    node: KDNode | None,
) -> _EChartsNode:

    if node is None:

        return {
            "name": "EMPTY",
            "children": []
        }

    text = ""

    if node.payload is not None:
        text = node.payload.text[:40]

    label = (
        f"id={node.item_id}\n"
        f"axis={node.axis}\n"
        f"{text}"
    )

    children = []

    if node.left is not None:
        children.append(node_to_echarts(node.left))

    if node.right is not None:
        children.append(node_to_echarts(node.right))

    return {
        "name": label,
        "children": children
    }

# Why: chart configuration should be isolated so layout fixes stay in one place.
def render_tree(tree: KDTreeIndex) -> None:
    """Render the semantic tree using ECharts."""
    data = node_to_echarts(tree.root)

    options = {
        "tooltip": {
            "trigger": "item",
            "triggerOn": "mousemove",
        },
        "series": [
            {
                "type": "tree",
                "data": [data],
                "top": "2%",
                "left": "10%",
                "bottom": "2%",
                "right": "28%",
                "symbol": "circle",
                "symbolSize": 10,
                "orient": "LR",
                "layout": "orthogonal",
                "expandAndCollapse": True,
                "initialTreeDepth": -1,
                "edgeShape": "polyline",
                "edgeForkPosition": "50%",
                "roam": True,
                "lineStyle": {
                    "width": 1.5,
                    "curveness": 0,
                },
                "label": {
                    "position": "left",
                    "verticalAlign": "middle",
                    "align": "right",
                    "fontSize": 12,
                    "lineHeight": 18,
                    "backgroundColor": "#f5f7fb",
                    "borderColor": "#d9e1ec",
                    "borderWidth": 1,
                    "borderRadius": 6,
                    "padding": [6, 8],
                },
                "leaves": {
                    "label": {
                        "position": "right",
                        "verticalAlign": "middle",
                        "align": "left",
                        "lineHeight": 18,
                        "backgroundColor": "#fff8e8",
                        "borderColor": "#f0d7a1",
                        "borderWidth": 1,
                        "borderRadius": 6,
                        "padding": [6, 8],
                    }
                },
                "animationDuration": 300,
                "animationDurationUpdate": 500
            }
        ],
    }

    st_echarts(options=options, height=900)


# Why: JSON loading is separated from UI flow to keep the script deterministic.
def load_records(path: Path) -> list[Record]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for item in raw:
        records.append(
            Record(id=item["id"], text=item["text"])
        )

    return records


# Why: initial records should be loaded once so local edits survive Streamlit reruns.
def initialize_records(file_path: Path) -> None:
    """Initialize editable records in session state."""
    if "records" not in st.session_state:
        st.session_state.records = load_records(file_path)


# Why: tree cache must be invalidated whenever the underlying dataset changes.
def invalidate_tree() -> None:
    """Drop cached tree metadata from session state."""
    st.session_state.pop("tree", None)
    st.session_state.pop("tree_key", None)


# Why: new ids should be assigned automatically from the current tree root maximum.
def get_next_record_id() -> int:
    """Return the next available record id."""
    tree: KDTreeIndex | None = st.session_state.get("tree")

    if tree is not None and tree.root.max_item_id is not None:
        return tree.root.max_item_id + 1

    records: list[Record] = st.session_state.records
    numeric_ids = [record.id for record in records]
    return max(numeric_ids, default=0) + 1


# Why: new nodes are created from user input and must keep ids unique and data consistent.
def add_record() -> None:
    """Add a new record into the editable dataset."""
    st.sidebar.subheader("Добавить запись")

    text = st.sidebar.text_area("Текст")
    if st.sidebar.button("Добавить"):
        if not text.strip():
            st.sidebar.error("Введите текст")
            return

        records = st.session_state.records

        max_id = max(r.id for r in records)

        records.append(
            Record(id=max_id + 1,text=text.strip())
        )

        st.session_state.records = records
        st.rerun()


# Why: deletion is implemented at dataset level because the tree is rebuilt after edits.
def delete_record() -> None:
    """Delete a record by id from the editable dataset."""
    st.sidebar.subheader("Удалить запись")
    record_id = st.sidebar.text_input("ID")

    if st.sidebar.button("Удалить"):
        if not record_id.strip():
            return

        records = st.session_state.records
        records = [
            r
            for r in records
            if str(r.id) != record_id
        ]

        st.session_state.records = records
        st.rerun()


# Why: the user benefits from seeing which ids are currently available for deletion and search.
def render_records_overview(records: list[Record]) -> None:
    """Render a compact overview of current record identifiers."""
    st.sidebar.caption(f"Всего записей: {len(records)}")
    st.sidebar.caption(
        f"Следующий id: {get_next_record_id()}" if records else "Следующий id: 1"
    )
    st.sidebar.caption(
        "Текущие id: " + ", ".join(str(record.id) for record in records[:12])
    )

    if len(records) > 12:
        st.sidebar.caption("Показаны первые 12 id.")


# Why: search results should be generated only when a valid tree exists.
def render_search_results(
    tree: KDTreeIndex | None,
    embedder: Embeddings
) -> None:
    """Render search results for the current query."""
    if tree is None:
        st.info("Нет данных для поиска. Добавьте хотя бы одну запись.")
        return

    st.subheader("Поиск")

    query = st.text_input(
        "Введите запрос",
        "космический корабль и спутники",
    )

    k = st.slider(
        "Top-K",
        1,
        20,
        5,
    )

    if st.button("Искать"):

        query_vector = normalize(
            embedder.embed(query)
        )

        results = tree.search(
            query=query_vector,
            k=k
        )

        st.subheader("Результаты")

        for result in results:
            st.markdown(
                f"""
                id={result.id}
                distance={result.distance:.4f}
                score={result.score:.4f}
                path={" -> ".join(result.path)}
                text={result.payload.text if result.payload else ""}
                """
            )


# Why: tree rendering should degrade gracefully when all records were deleted.
def render_tree_panel(tree: KDTreeIndex | None) -> None:
    """Render the tree panel or an empty-state message."""
    st.subheader("Дерево")

    if tree is None:
        st.info("Дерево пустое. Добавьте записи, чтобы построить его снова.")
        return

    render_tree(tree)


# Why: the app entrypoint keeps Streamlit layout and interactions explicit.
def main() -> None:
    """Run the Streamlit demo application."""
    st.set_page_config(layout="wide")
    st.title("Semantic KD-Tree")

    data_path = Path(__file__).with_name("initial_data.json")

    if "records" not in st.session_state:
        st.session_state.records = load_records(
            data_path
        )

    initialize_records(data_path)
    embedder = Embeddings()

    add_record()
    delete_record()

    records: list[Record] = st.session_state.records
    render_records_overview(records)

    st.sidebar.markdown(
        f"Всего записей: {len(records)}"
    )

    with st.spinner(
            "Building KD-tree:"
    ):
        tree = build_tree(
            records=records,
            embedder=embedder
        )

    left, right = st.columns([1, 2])

    with left:
        render_search_results(
            tree=tree,
            embedder=embedder
        )

    with right:
        render_tree_panel(tree)


if __name__ == "__main__":
    main()
