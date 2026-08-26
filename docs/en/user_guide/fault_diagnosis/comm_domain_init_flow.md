# Overall Process

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-07-31T03:57:33.243Z pushedAt=2026-08-13T06:37:32.544Z -->

The communicator initialization process is shown in the following figure:

![](figures/comm_domain_init_flow.png "Communicator initialization process")

During the communicator initialization phase, HCCL goes through multiple processes as shown in the preceding figure. First, environment variable initialization and resource initialization are performed, and then the cluster information of the entire communicator is obtained. There are generally two methods for this process:

- **Creating a communicator based on a rank table file**: A rank table (cluster information configuration file) is generated through other means, and the HCCL interface for creating a communicator is called to read the corresponding file. For the format requirements of the rank table itself, see [Cluster Information Configuration](../cluster_info_config/README.md).

    When configuring communicator information through a rank table file, ensure that the file path and permissions are correct, and that the file remains consistent across all ranks in the cluster. HCCL performs rank table consistency checks among ranks during the subsequent parameter plane link establishment of operators, and terminates the service if the requirements are not met.

- **Creating a communicator based on root node information**: Also referred to as creating a communicator through cluster negotiation. This method establishes a socket connection to the root node via the Host-side NIC through the communicator creation interface provided by HCCL, thereby aggregating and distributing information to generate cluster information.

When creating a communicator based on root node information, ensure that the configured NIC and port are correct. Additionally, if a fault causes some ranks to fail to deliver information to the root node in a timely manner, this stage will also fail.

Regardless of which method is used to create the communicator, HCCL ultimately verifies the generated cluster information, checking whether the cluster hardware configuration is abnormal, such as duplicate IP addresses, mixed use of IPv4 and IPv6, and inconsistent TLS configurations.
