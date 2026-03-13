import importlib
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Optional, Tuple

import gradio as gr
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import BaseMessage, add_messages
from typing_extensions import TypedDict


UDP_HOST = "127.0.0.1"
UDP_PORT = 20777
STALE_SECONDS = 10.0
SESSION_TYPE_MAP = {
    0: "Unknown",
    1: "P1",
    2: "P2",
    3: "P3",
    4: "Short P",
    5: "Q1",
    6: "Q2",
    7: "Q3",
    8: "Short Q",
    9: "OSQ",
    10: "Race",
    11: "Race 2",
    12: "Time Trial",
}
WEATHER_MAP = {
    0: "Clear",
    1: "Light Cloud",
    2: "Overcast",
    3: "Light Rain",
    4: "Heavy Rain",
    5: "Storm",
}
SAFETY_CAR_STATUS_MAP = {
    0: "None",
    1: "Safety Car",
    2: "Virtual Safety Car",
    3: "Formation Lap Safety Car",
}
TYRE_COMPOUND_MAP = {
    7: "Intermediate",
    8: "Wet",
    16: "C5",
    17: "C4",
    18: "C3",
    19: "C2",
    20: "C1",
}
TYRE_VISUAL_MAP = {
    7: "Intermediate",
    8: "Wet",
    16: "Soft",
    17: "Medium",
    18: "Hard",
}


load_dotenv()
llm = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai"
)


@dataclass
class TelemetrySnapshot:
    received_at: float
    session_type: Optional[str] = None
    lap: Optional[int] = None
    tyre_compound: Optional[str] = None
    fuel_kg: Optional[float] = None
    position: Optional[int] = None
    raw_packet_id: Optional[int] = None
    raw_bytes: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


class TelemetryStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot: Optional[TelemetrySnapshot] = None

    def update(self, snapshot: TelemetrySnapshot) -> None:
        with self._lock:
            if self._snapshot is None:
                self._snapshot = snapshot
                return
            current = self._snapshot
            self._snapshot = TelemetrySnapshot(
                received_at=snapshot.received_at,
                session_type=snapshot.session_type or current.session_type,
                lap=snapshot.lap if snapshot.lap is not None else current.lap,
                tyre_compound=snapshot.tyre_compound or current.tyre_compound,
                fuel_kg=snapshot.fuel_kg if snapshot.fuel_kg is not None else current.fuel_kg,
                position=snapshot.position if snapshot.position is not None else current.position,
                raw_packet_id=snapshot.raw_packet_id if snapshot.raw_packet_id is not None else current.raw_packet_id,
                raw_bytes=snapshot.raw_bytes or current.raw_bytes,
                extra={**current.extra, **snapshot.extra},
            )

    def get(self) -> Optional[TelemetrySnapshot]:
        with self._lock:
            return self._snapshot


class TelemetryParser:
    def __init__(self) -> None:
        self._parse_fn = None
        self.status = "missing"
        self.error: Optional[str] = None
        self._init_parser()

    def _init_parser(self) -> None:
        try:
            module = importlib.import_module("f1_2020_telemetry.packets")
            attr = getattr(module, "unpack_udp_packet", None)
            if callable(attr):
                self._parse_fn = attr
                self.status = "ok:f1_2020_telemetry.packets.unpack_udp_packet"
                return
        except Exception as exc:  # pragma: no cover - best effort import
            self.error = f"{type(exc).__name__}: {exc}"

        self.status = "missing"

    def parse(self, data: bytes) -> TelemetrySnapshot:
        raw_bytes = len(data)
        received_at = time.time()
        if self._parse_fn is None:
            return TelemetrySnapshot(
                received_at=received_at,
                raw_bytes=raw_bytes,
                extra={"parser_status": "missing", "error": self.error},
            )

        try:
            parsed = self._parse_fn(data)
        except Exception as exc:
            return TelemetrySnapshot(
                received_at=received_at,
                raw_bytes=raw_bytes,
                extra={"parser_status": "error", "error": f"{type(exc).__name__}: {exc}"},
            )

        return _build_snapshot(parsed, raw_bytes, received_at, self.status)


def _build_snapshot(parsed: Any, raw_bytes: int, received_at: float, parser_status: str) -> TelemetrySnapshot:
    data: Dict[str, Any] = {}
    if isinstance(parsed, dict):
        data = parsed
    else:
        if hasattr(parsed, "header"):
            header = getattr(parsed, "header")
            data["packet_id"] = getattr(header, "packetId", None)
            data["player_car_index"] = getattr(header, "playerCarIndex", None)
        _hydrate_from_packet(parsed, data)
        for attr in [
            "session_type",
            "sessionType",
            "m_sessionType",
            "lap",
            "current_lap",
            "m_currentLapNum",
            "tyre_compound",
            "tyreCompound",
            "m_tyreCompound",
            "fuel_in_tank",
            "fuelInTank",
            "m_fuelInTank",
            "position",
            "m_carPosition",
            "packet_id",
            "m_packetId",
        ]:
            if hasattr(parsed, attr):
                data[attr] = getattr(parsed, attr)

    snapshot = TelemetrySnapshot(received_at=received_at, raw_bytes=raw_bytes)
    snapshot.session_type = _first_present(
        data,
        ["session_type", "sessionType", "m_sessionType"],
    )
    if snapshot.session_type is not None:
        try:
            session_id = int(snapshot.session_type)
        except Exception:
            session_id = None
        if session_id is not None:
            data["session_type_id"] = session_id
            snapshot.session_type = SESSION_TYPE_MAP.get(session_id, str(session_id))
    snapshot.lap = _first_present(
        data,
        ["lap", "current_lap", "currentLapNum", "m_currentLapNum"],
        cast=int,
    )
    snapshot.tyre_compound = _first_present(
        data,
        ["tyre_compound", "tyreCompound", "m_tyreCompound", "actualTyreCompound", "visualTyreCompound"],
    )
    if snapshot.tyre_compound is not None:
        try:
            compound_id = int(snapshot.tyre_compound)
        except Exception:
            compound_id = None
        if compound_id is not None:
            actual_name = TYRE_COMPOUND_MAP.get(compound_id)
            visual_name = TYRE_VISUAL_MAP.get(compound_id)
            snapshot.tyre_compound = actual_name or visual_name or str(compound_id)
    snapshot.fuel_kg = _first_present(
        data,
        ["fuel_in_tank", "fuelInTank", "m_fuelInTank"],
        cast=float,
    )
    snapshot.position = _first_present(
        data,
        ["position", "carPosition", "m_carPosition"],
        cast=int,
    )
    snapshot.raw_packet_id = _first_present(
        data,
        ["packet_id", "m_packetId"],
        cast=int,
    )
    extra = {}
    for key, value in list(data.items())[:20]:
        extra[key] = value
    for key in ["all_total_distance", "all_positions", "all_speed", "totalDistance"]:
        if key in data:
            extra[key] = data[key]
    if "player_car_index" in data:
        extra["player_car_index"] = data["player_car_index"]
    extra["parser_status"] = parser_status
    if "session_type_id" in data:
        extra["session_type_id"] = data["session_type_id"]
    snapshot.extra = extra
    return snapshot


def _hydrate_from_packet(parsed: Any, data: Dict[str, Any]) -> None:
    try:
        from f1_2020_telemetry import packets
    except Exception:
        return

    def _get_by_index(collection: Any, index: Optional[int]) -> Optional[Any]:
        if collection is None or index is None:
            return None
        try:
            return collection[index]
        except Exception:
            return None

    if isinstance(parsed, packets.PacketSessionData_V1):
        data["sessionType"] = getattr(parsed, "sessionType", None)
        data["weather"] = getattr(parsed, "weather", None)
        data["safetyCarStatus"] = getattr(parsed, "safetyCarStatus", None)
        data["totalLaps"] = getattr(parsed, "totalLaps", None)
        data["trackLength"] = getattr(parsed, "trackLength", None)
        return

    if isinstance(parsed, packets.PacketLapData_V1):
        header = getattr(parsed, "header", None)
        index = data.get("player_car_index")
        if index is None and header is not None:
            index = getattr(header, "playerCarIndex", None)
        all_lap = getattr(parsed, "lapData", None)
        if all_lap is not None:
            data["all_total_distance"] = [getattr(item, "totalDistance", None) for item in all_lap]
            data["all_positions"] = [getattr(item, "carPosition", None) for item in all_lap]
        car = _get_by_index(getattr(parsed, "lapData", None), index)
        if car is not None:
            data["currentLapNum"] = getattr(car, "currentLapNum", None)
            data["carPosition"] = getattr(car, "carPosition", None)
            data["safetyCarDelta"] = getattr(car, "safetyCarDelta", None)
            data["lastLapTime"] = getattr(car, "lastLapTime", None)
            data["totalDistance"] = getattr(car, "totalDistance", None)
        return

    if isinstance(parsed, packets.PacketCarStatusData_V1):
        header = getattr(parsed, "header", None)
        index = data.get("player_car_index")
        if index is None and header is not None:
            index = getattr(header, "playerCarIndex", None)
        car = _get_by_index(getattr(parsed, "carStatusData", None), index)
        if car is not None:
            data["fuelInTank"] = getattr(car, "fuelInTank", None)
            data["actualTyreCompound"] = getattr(car, "actualTyreCompound", None)
            data["visualTyreCompound"] = getattr(car, "visualTyreCompound", None)
            data["tyresAgeLaps"] = getattr(car, "tyresAgeLaps", None)
            data["tyresWear"] = getattr(car, "tyresWear", None)
            data["ersStoreEnergy"] = getattr(car, "ersStoreEnergy", None)
            data["ersDeployMode"] = getattr(car, "ersDeployMode", None)
            data["fuelRemainingLaps"] = getattr(car, "fuelRemainingLaps", None)
        return

    if isinstance(parsed, packets.PacketCarTelemetryData_V1):
        header = getattr(parsed, "header", None)
        index = data.get("player_car_index")
        if index is None and header is not None:
            index = getattr(header, "playerCarIndex", None)
        all_tel = getattr(parsed, "carTelemetryData", None)
        if all_tel is not None:
            data["all_speed"] = [getattr(item, "speed", None) for item in all_tel]
        car = _get_by_index(getattr(parsed, "carTelemetryData", None), index)
        if car is not None:
            data["drs"] = getattr(car, "drs", None)
            data["speed"] = getattr(car, "speed", None)
        return


def _first_present(data: Dict[str, Any], keys: List[str], cast=None):
    for key in keys:
        if key in data and data[key] is not None:
            value = data[key]
            if cast is not None:
                try:
                    return cast(value)
                except Exception:
                    return value
            return value
    return None


def _compute_gaps(snapshot: TelemetrySnapshot) -> Dict[str, Optional[float]]:
    distances = snapshot.extra.get("all_total_distance")
    positions = snapshot.extra.get("all_positions")
    speeds = snapshot.extra.get("all_speed")
    player_index = snapshot.extra.get("player_car_index")
    player_distance = snapshot.extra.get("totalDistance")
    player_speed = snapshot.extra.get("speed")

    if not isinstance(distances, list) or player_distance is None:
        return {}
    try:
        player_distance = float(player_distance)
    except Exception:
        return {}

    ahead_index = None
    behind_index = None

    if isinstance(positions, list) and snapshot.position is not None:
        try:
            target_ahead = snapshot.position - 1
            target_behind = snapshot.position + 1
            ahead_index = positions.index(target_ahead) if target_ahead in positions else None
            behind_index = positions.index(target_behind) if target_behind in positions else None
        except Exception:
            ahead_index = None
            behind_index = None

    if ahead_index is None or behind_index is None:
        indexed = []
        for i, dist in enumerate(distances):
            if dist is None:
                continue
            try:
                indexed.append((i, float(dist)))
            except Exception:
                continue
        indexed.sort(key=lambda item: item[1])
        if indexed:
            player_idx = None
            for i, dist in indexed:
                if i == player_index:
                    player_idx = i
                    player_distance = dist
                    break
            if player_idx is None:
                player_idx = indexed[-1][0]
            ahead_index = None
            behind_index = None
            for i, dist in indexed:
                if dist > player_distance:
                    ahead_index = i
                    break
            for i, dist in reversed(indexed):
                if dist < player_distance:
                    behind_index = i
                    break

    gap_ahead_m = None
    gap_behind_m = None
    if ahead_index is not None and ahead_index < len(distances):
        dist = distances[ahead_index]
        if dist is not None:
            try:
                gap_ahead_m = float(dist) - player_distance
            except Exception:
                pass
    if behind_index is not None and behind_index < len(distances):
        dist = distances[behind_index]
        if dist is not None:
            try:
                gap_behind_m = player_distance - float(dist)
            except Exception:
                pass

    gap_ahead_s = None
    gap_behind_s = None
    if player_speed is not None:
        try:
            speed_mps = float(player_speed) / 3.6
        except Exception:
            speed_mps = None
        if speed_mps and speed_mps > 0:
            if gap_ahead_m is not None:
                gap_ahead_s = gap_ahead_m / speed_mps
            if gap_behind_m is not None:
                gap_behind_s = gap_behind_m / speed_mps

    return {
        "gap_ahead_m": gap_ahead_m,
        "gap_behind_m": gap_behind_m,
        "gap_ahead_s": gap_ahead_s,
        "gap_behind_s": gap_behind_s,
    }


class UdpListener(threading.Thread):
    def __init__(self, host: str, port: int, store: TelemetryStore, parser: TelemetryParser) -> None:
        super().__init__(daemon=True)
        self._host = host
        self._port = port
        self._store = store
        self._parser = parser
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self._host, self._port))
        sock.settimeout(1.0)
        while not self._stop_event.is_set():
            try:
                data, _addr = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except Exception:
                continue
            snapshot = self._parser.parse(data)
            self._store.update(snapshot)


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


telemetry_store = TelemetryStore()
telemetry_parser = TelemetryParser()


def _telemetry_context() -> str:
    snapshot = telemetry_store.get()
    if snapshot is None:
        return "No telemetry received yet."

    age = time.time() - snapshot.received_at
    stale_note = "stale" if age > STALE_SECONDS else "fresh"
    gap_info = _compute_gaps(snapshot)
    tyres_wear = snapshot.extra.get("tyresWear")
    tyre_wear_text = None
    if isinstance(tyres_wear, (list, tuple)) and len(tyres_wear) >= 4:
        tyre_wear_text = f"FL={tyres_wear[0]}, FR={tyres_wear[1]}, RL={tyres_wear[2]}, RR={tyres_wear[3]}"
    ers_energy = snapshot.extra.get("ersStoreEnergy")
    ers_mode = snapshot.extra.get("ersDeployMode")
    weather = snapshot.extra.get("weather")
    safety_car = snapshot.extra.get("safetyCarStatus")
    if weather is not None:
        try:
            weather = WEATHER_MAP.get(int(weather), weather)
        except Exception:
            pass
    if safety_car is not None:
        try:
            safety_car = SAFETY_CAR_STATUS_MAP.get(int(safety_car), safety_car)
        except Exception:
            pass
    parts = [
        f"status={stale_note}",
        f"age_sec={age:.1f}",
        f"session_type={snapshot.session_type}",
        f"lap={snapshot.lap}",
        f"total_laps={snapshot.extra.get('totalLaps')}",
        f"tyre_compound={snapshot.tyre_compound}",
        f"tyre_wear={tyre_wear_text}",
        f"fuel_kg={snapshot.fuel_kg}",
        f"fuel_laps={snapshot.extra.get('fuelRemainingLaps')}",
        f"weather={weather}",
        f"safety_car={safety_car}",
        f"position={snapshot.position}",
        f"gap_ahead_m={gap_info.get('gap_ahead_m')}",
        f"gap_ahead_s={gap_info.get('gap_ahead_s')}",
        f"gap_behind_m={gap_info.get('gap_behind_m')}",
        f"gap_behind_s={gap_info.get('gap_behind_s')}",
        f"safety_car_delta={snapshot.extra.get('safetyCarDelta')}",
        f"speed_kph={snapshot.extra.get('speed')}",
        f"drs={snapshot.extra.get('drs')}",
        f"ers_energy={ers_energy}",
        f"ers_mode={ers_mode}",
        f"packet_id={snapshot.raw_packet_id}",
        f"parser={snapshot.extra.get('parser_status')}",
        f"raw_bytes={snapshot.raw_bytes}",
    ]
    return "Telemetry: " + ", ".join(parts)


def f1_advisor_agent(state: State):
    telemetry_text = _telemetry_context()
    system_prompt = (
        "You are an F1 2020 race engineer and strategy advisor. "
        "Use the provided telemetry context to answer questions for practice, qualifying, or race. "
        "If telemetry is missing or stale, ask for the missing context and explain what you need. "
        "Give clear, actionable strategy guidance. "
        "Do not mention being a therapist or provide emotional support.\n\n"
        "Telemetry meaning guide:\n"
        "- fuel_laps: treat this as excess laps above race distance (not total remaining laps).\n"
        "- fuel_kg: current fuel mass in kg.\n"
        "- session_type: session identifier (P/Q/R) derived from sessionType.\n"
        "- lap/total_laps: current lap number and total race laps.\n"
        "- tyre_compound: tyre compound name derived from compound ID.\n"
        "- tyre_wear: per-tyre wear percentages (FL/FR/RL/RR) from car status.\n"
        "- safety_car: safety car status derived from session data.\n"
        "- weather: track weather status derived from session data.\n"
        "- ers_energy: ERS stored energy (Joules) from car status.\n"
        "- ers_mode: ERS deploy mode from car status.\n"
        "- safety_car_delta: delta to safety car, not gap to car ahead.\n"
        "- gap_ahead/behind: estimated from totalDistance and speed, may be approximate.\n\n"
        f"{telemetry_text}"
    )
    messages_with_prompt = [SystemMessage(content=system_prompt)] + state["messages"]
    reply = llm.invoke(messages_with_prompt)
    return {"messages": [AIMessage(content=reply.content)]}


graph_builder = StateGraph(State)
graph_builder.add_node("f1_advisor", f1_advisor_agent)
graph_builder.add_edge(START, "f1_advisor")
graph_builder.add_edge("f1_advisor", END)
graph = graph_builder.compile()


def _append_history(messages_list: List[BaseMessage], history_item: Any) -> None:
    if isinstance(history_item, (list, tuple)) and len(history_item) == 2:
        human_msg, ai_msg = history_item
        messages_list.append(HumanMessage(content=str(human_msg)))
        messages_list.append(AIMessage(content=str(ai_msg)))
        return
    if isinstance(history_item, dict):
        role = history_item.get("role")
        content = history_item.get("content", "")
        if role == "user":
            messages_list.append(HumanMessage(content=str(content)))
            return
        if role in ("assistant", "bot"):
            messages_list.append(AIMessage(content=str(content)))
            return
    if isinstance(history_item, str):
        messages_list.append(HumanMessage(content=history_item))
        return


def chatbot_interface(message: str, history: List[Any]):
    messages_list: List[BaseMessage] = []
    for item in history:
        _append_history(messages_list, item)
    messages_list.append(HumanMessage(content=message))
    state = {"messages": messages_list}
    response = graph.invoke(state)
    ai_message = response["messages"][-1]
    return ai_message.content


demo = gr.ChatInterface(
    fn=chatbot_interface,
    title="F1 UDP Advisor",
    description="Ask for practice, qualifying, or race strategy using live F1 2020 telemetry.",
    examples=[
        ["Practice plan for a new track with soft tires."],
        ["Quali strategy for Q1 and Q2 with current tire wear."],
        ["Race strategy for pit windows and tire choices."],
    ],
)


if __name__ == "__main__":
    listener = UdpListener(UDP_HOST, UDP_PORT, telemetry_store, telemetry_parser)
    listener.start()
    demo.launch()
