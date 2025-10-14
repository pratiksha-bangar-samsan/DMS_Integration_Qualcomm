
import os
import can
 
def initialize_pcan(interface='can0', bitrate=500000):
    """
    Initialize and verify PCAN-USB (SocketCAN interface) on Linux.
    """
    print(" Checking available CAN interfaces...")
    os.system("ip link show")
 
    # Bring up the CAN interface with the specified bitrate
    print(f"\nSetting up {interface} with bitrate {bitrate}...")
    os.system(f"sudo ip link set {interface} down")
    os.system(f"sudo ip link set {interface} type can bitrate {bitrate}")
    os.system(f"sudo ip link set {interface} up")
 
    # Verify interface status
    print("\n Checking if interface is up...")
    os.system(f"ip link show {interface}")
 
    try:
        print(f"\n Initializing CAN bus on {interface}...")
        bus = can.Bus(channel=interface, interface='socketcan')
        print("PCAN initialized successfully!")
        return bus
    except Exception as e:
        print(f" Failed to initialize CAN interface: {e}")
        return None
 
 
def test_can(bus):
    """
    Send and receive a test CAN frame.
    """
    if not bus:
        print("No CAN bus initialized.")
        return
 
    msg = can.Message(
        arbitration_id=0x123,
        data=[0x11, 0x22, 0x33, 0x44],
        is_extended_id=False
    )
 
    try:
        print(" Sending CAN message...")
        bus.send(msg)
        print(" Message sent successfully!")
 
        print("📥 Waiting for a response (3s timeout)...")
        message = bus.recv(3.0)
        if message is None:
            print("️ No message received within timeout.")
        else:
            print(f"✅ Received message: {message}")
    except can.CanError as e:
        print(f"CAN transmission error: {e}")
 
 
if __name__ == "__main__":
    bus = initialize_pcan(interface='can0', bitrate=500000)
    try:
        test_can(bus)
    finally:
        if bus:
            print("\n🧹 Closing CAN bus...")
            bus.shutdown()
            print("CAN bus closed properly.")
