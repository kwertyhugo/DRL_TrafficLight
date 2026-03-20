import xml.etree.ElementTree as ET
import statistics

def analyze_time_loss(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # -----------------------------
    # VEHICLE DATA
    # -----------------------------
    vehicle_time_loss = []

    for trip in root.findall("tripinfo"):
        if "timeLoss" in trip.attrib:
            vehicle_time_loss.append(float(trip.attrib["timeLoss"]))

    # -----------------------------
    # PEDESTRIAN DATA
    # -----------------------------
    ped_total_loss = []
    ped_waiting_loss = []
    ped_movement_loss = []

    for person in root.findall("personinfo"):
        total_loss = float(person.attrib.get("timeLoss", 0.0))
        waiting = float(person.attrib.get("waitingTime", 0.0))
        movement = total_loss - waiting

        ped_total_loss.append(total_loss)
        ped_waiting_loss.append(waiting)
        ped_movement_loss.append(movement)

    # -----------------------------
    # PRINT FUNCTION
    # -----------------------------
    def print_stats(title, data):
        if len(data) < 2:
            print(f"{title}: Not enough data")
            return
        print(f"\n{title}")
        print(f"Count: {len(data)}")
        print(f"Mean: {statistics.mean(data):.3f}")
        print(f"Std Dev: {statistics.stdev(data):.3f}")

    # -----------------------------
    # OUTPUT
    # -----------------------------
    print("\n================ TIME LOSS ANALYSIS ================")

    print_stats("Vehicles – Time Loss", vehicle_time_loss)

    print_stats("Pedestrians – Total Time Loss", ped_total_loss)
    print_stats("Pedestrians – Waiting Time Loss", ped_waiting_loss)
    print_stats("Pedestrians – Movement Time Loss", ped_movement_loss)

    print("\n====================================================")

# -----------------------------
# RUN HERE
# -----------------------------
analyze_time_loss(
    "Banlic-Mamatid_traci/output_DDPG/test_jam_trips.xml"
)