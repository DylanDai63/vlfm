import socket
import json
import time
import numpy as np
from pathlib import Path


def softmax(logits):
    """Compute softmax probabilities from logits"""
    logits = np.array(logits)
    exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
    return exp_logits / exp_logits.sum()


class DirectionClient:
    def __init__(self, host='171.248.40.12', port=50500):
        self.host = host
        self.port = port
        self.sock = None

    def connect(self):
        """Connect to the server"""
        print(f"Connecting to {self.host}:{self.port}...")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        print(f"✅ Connected to {self.host}:{self.port}")

    def send_image(self, image_path):
        """Send image and receive logits"""
        # Read image file
        with open(image_path, 'rb') as f:
            image_data = f.read()

        # Send image size
        self.sock.sendall(len(image_data).to_bytes(8, 'big'))

        # Send image data
        self.sock.sendall(image_data)

        # Receive result size
        size_data = self.sock.recv(8)
        result_size = int.from_bytes(size_data, 'big')

        # Receive result
        result_data = b''
        while len(result_data) < result_size:
            chunk = self.sock.recv(min(result_size - len(result_data), 4096))
            result_data += chunk

        result = json.loads(result_data.decode('utf-8'))
        return result

    def close(self):
        """Close connection"""
        if self.sock:
            self.sock.close()
            print("Connection closed")


if __name__ == "__main__":
    # Configuration
    IMAGE_PATH = "test_image.png"  # ⚠️ CHANGE THIS to your actual image file!
    SEND_DELAY = 0.5  # Delay between sends in seconds (adjust as needed)

    # client = DirectionClient(host='171.248.40.12', port=58893)  # For direct connection
    client = DirectionClient(host='localhost', port=9999)  # For SSH tunnel

    try:
        client.connect()

        print(f"\n🔄 Starting continuous feed (sending {IMAGE_PATH} repeatedly)")
        print(f"   Delay between sends: {SEND_DELAY}s")
        print("   Press Ctrl+C to stop\n")

        image_count = 0

        while True:
            image_count += 1

            # Send image and get result
            print(f"{'=' * 60}")
            print(f"🚀 Sending image #{image_count}...")
            result = client.send_image(IMAGE_PATH)

            # Extract logits from server
            logits = result['logits']

            # Compute probabilities locally using softmax
            probs = softmax(logits)

            best_choice = ['A', 'B', 'C', 'D'][np.argmax(probs)]
            confidence = np.max(probs)

            print(f"📊 Probabilities: A:{probs[0]:.2%} B:{probs[1]:.2%} C:{probs[2]:.2%} D:{probs[3]:.2%}")
            print(f"✅ Best: {best_choice} ({confidence:.1%} confidence)")
            print(f"⏱️  Waiting {SEND_DELAY}s before next send...\n")

            time.sleep(SEND_DELAY)

    except KeyboardInterrupt:
        print(f"\n\n⏹️  Stopped by user after {image_count} images")
    except ConnectionRefusedError:
        print("\n❌ ERROR: Cannot connect to server!")
        print("   Check that:")
        print("   1. Server is running on vast.ai")
        print("   2. SSH tunnel is active: ssh -p 50435 root@171.248.40.12 -L 9999:localhost:5555")
    except FileNotFoundError:
        print(f"\n❌ ERROR: Image file '{IMAGE_PATH}' not found!")
        print("   Please place an image file in the same folder and update IMAGE_PATH")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
    finally:
        client.close()