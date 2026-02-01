# src/app/azure_listener.py
"""
Azure IoT Hub Listener Service.

This script acts as the 'Backend' or 'Producer' in the Producer-Consumer architecture.
It maintains a persistent connection to the Azure Event Hub, listens for incoming
telemetry messages from the ESP32, and safely writes them to a shared JSON file.

Architecture Benefit:
By decoupling data fetching (this script) from data visualization (Streamlit),
we prevent the UI from freezing while waiting for network packets.
"""

import os
import json
import sys
from datetime import datetime, timedelta, timezone
from azure.eventhub import EventHubConsumerClient
from dotenv import load_dotenv

# =============================================================================
#  CONFIGURATION
# =============================================================================
# Load environment variables from the .env file (if present)
load_dotenv()

# Retrieve the Connection String
CONNECTION_STR = os.getenv("AZURE_IOTHUB_CONNECTION_STRING")

# Validation: Stop execution immediately if the key is missing
if not CONNECTION_STR:
    print("❌ CRITICAL ERROR: 'AZURE_IOTHUB_CONNECTION_STRING' not found.")
    print("Please create a .env file in the project root with this variable.")
    sys.exit(1)

# The Consumer Group allows multiple applications to read the same stream independently.
CONSUMER_GROUP = "python_dashboard"

# The shared file acting as the database/buffer between this script and the Dashboard.
OUTPUT_FILE = "latest_telemetry.json"

def save_data_atomically(data: dict) -> None:
    """
    Saves data to disk using the 'Atomic Write' pattern.

    Why is this important?
    If we write directly to 'latest_telemetry.json', the Streamlit dashboard might
    try to read the file *while* we are writing to it, resulting in a corrupted
    JSON error (half-written file).

    Solution:
    1. Write to a temporary file (e.g., .tmp).
    2. Rename the temp file to the final filename.

    The OS guarantees that the 'rename' operation is atomic (instantaneous),
    so the Dashboard never sees a corrupt file.
    """
    try:
        # Inject a local timestamp for debugging latency issues
        data['_local_saved_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        temp_file = OUTPUT_FILE + ".tmp"

        # Write to the temp file
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=4)

        # Atomic replace
        # On Windows, os.rename cannot overwrite existing files, so we remove first.
        if os.path.exists(OUTPUT_FILE):
            os.remove(OUTPUT_FILE)

        os.rename(temp_file, OUTPUT_FILE)

        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Data persisted to {OUTPUT_FILE}")

    except Exception as e:
        print(f"❌ Critical Error saving file: {e}")


def on_event_received(partition_context, event):
    """
    Callback function triggered by Azure whenever a new message arrives.

    Args:
        partition_context: Metadata about the Event Hub partition.
        event: The actual data object containing the payload.
    """
    try:
        # Azure sends the body as bytes, and I need to decode it to string
        body = event.body_as_str(encoding='UTF-8')
        payload = json.loads(body)

        # --- DATA NORMALIZATION LOGIC ---
        # Azure IoT Hub sometimes wraps the message in a "body" key,
        # or sends it flat, depending on how the ESP32 sent it.
        # We need to extract the actual telemetry data reliably.
        target_data = None

        # Scenario A: Wrapped inside "body" object
        if "body" in payload and isinstance(payload["body"], dict) and "history" in payload["body"]:
            target_data = payload["body"]

        # Scenario B: Flat structure (Direct telemetry)
        elif "history" in payload:
            target_data = payload

        # Process if valid data found
        if target_data:
            save_data_atomically(target_data)

            # Checkpoint the progress.
            # This tells Azure: "I have successfully processed this message."
            # If the script crashes, it will resume from here, not from the beginning.
            partition_context.update_checkpoint(event)
        else:
            print(f"⚠️ Warning: Received message with invalid format (missing 'history').")

    except json.JSONDecodeError:
        print(f"❌ Error: Received non-JSON data from Azure.")
    except Exception as e:
        print(f"❌ Unexpected Error processing event: {e}")

def main():
    """
        Main entry point. Initializes the connection and starts the blocking listener loop.
    """
    print("=" * 50)
    print("📡 AZURE IOT HUB LISTENER SERVICE")
    print("=" * 50)
    print(f"Target File: {os.path.abspath(OUTPUT_FILE)}")
    print(f"Consumer Group: {CONSUMER_GROUP}")
    print("Status: Connecting to Azure Cloud...")

    # I start listening from 6 hours ago to ensure we catch any recent data
    # if the script was offline for a while.
    start_time = datetime.now(timezone.utc) - timedelta(hours=6)

    try:
        client = EventHubConsumerClient.from_connection_string(
            conn_str=CONNECTION_STR,
            consumer_group=CONSUMER_GROUP,
            eventhub_name=None  # Automatically inferred from the connection string
        )
    except Exception as e:
        print(f"❌ Connection Error: Could not create client. Check connection string.\n{e}")
        sys.exit(1)

    print("✅ Connected! Waiting for incoming telemetry...")

    with client:
        # client.receive is a BLOCKING call.
        # It keeps the script running indefinitely, waiting for events.
        try:
            client.receive(
                on_event=on_event_received,
                starting_position=start_time
            )
        except KeyboardInterrupt:
            print("\n🛑 Stopping listener service...")

if __name__ == "__main__":
    main()