import os

try:
    if not os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", None):
        # Fallback to local endpoint 
        os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"

    from phoenix.otel import register
        # Configuring phoenix tracing, for all llm calls 
    tracer_provider = register(
        project_name="hybrid-rag-app",
        auto_instrument=True # Auto-instrument your app based on installed OI dependencies
    )

    print("INFO: Tracing enabled.")
except Exception as e:
    print(f"WARNING: Not able to enable Tracing: {e}")
    # Handle the error or log it as needed