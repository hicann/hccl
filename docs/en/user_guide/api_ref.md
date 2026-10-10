# API Reference

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-02T07:59:56.601Z pushedAt=2026-09-07T10:00:44.445Z -->

HCCL provides communicator management APIs and communication operator APIs, supporting framework developers in implementing distributed capabilities.

- Communicator management APIs: provide APIs for creating, destroying, and handling exceptions of communicators.

  Communicator management APIs support both C and Python. The C APIs are used to implement framework adaptation in single operator mode and enable distributed capabilities. The Python APIs are used to implement framework adaptation in graph mode and are currently used only for distributed optimization of TensorFlow networks on NPUs.
- Communication operator APIs: provide two categories of APIs: collective communication and point-to-point communication.
