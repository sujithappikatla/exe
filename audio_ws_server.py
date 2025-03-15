import os
os.environ['GRPC_PYTHON_LOG_LEVEL'] = '5'  # Add this line before other imports
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./stt-service-account.json"

from google.cloud import speech


import asyncio
import websockets

# Set to hold connected clients
connected_clients = set()


async def audio_handler(websocket):
    # Register the new client
    connected_clients.add(websocket)
    print("Client connected")

     # Initialize the Speech client
    client = speech.SpeechClient()

    # Configure recognition parameters
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,  # To get results as you speak
    )

    try:
        requests = []  # List to hold StreamingRecognizeRequest objects
        async for message in websocket:
            # Process incoming audio data (binary data)
            # print("Received audio data of length:", len(message))
            # Here you could save the audio data or process it as needed

            request = speech.StreamingRecognizeRequest(audio_content=message)
            requests.append(request)

            # Send the requests manually in batches to the API
            if len(requests) >= 20:  # Send in batches of 10
                responses = client.streaming_recognize(
                    config=streaming_config, requests=iter(requests)
                )
                for response in responses:
                    for result in response.results:
                        if result.is_final:
                            print(f"Final: {result.alternatives[0].transcript}")
                            await websocket.send(f"Final: {result.alternatives[0].transcript}")
                        # else:
                            # print(f"Interim: {result.alternatives[0].transcript}", end="\r")
                            # await websocket.send(f"Interim: {result.alternatives[0].transcript}")
                requests.clear()  # Clear the batch after sending
            
            # Optionally, send an acknowledgment back to the client
            # await websocket.send("Audio data received")
    except websockets.exceptions.ConnectionClosed as e:
        print("Client disconnected:", e)
    finally:
        # Unregister the client
        connected_clients.remove(websocket)

# Start the WebSocket server
async def main():
    server = await websockets.serve(audio_handler, "localhost", 8765)
    print("WebSocket server started on ws://localhost:8765")
    await server.wait_closed()

# Run the server
if __name__ == "__main__":
    asyncio.run(main())