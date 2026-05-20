# 集群信息校验失败问题

## 定位思路

HCCL会对rank table文件或协商收集到的rank table信息进行校验，若校验失败HCCL会直接报错退出，请基于实际报错内容进行定位。

可能的原因有：rank table文件校验失败、内容与硬件配置不符合，TLS配置不一致或superDeviceId重复。

后续内容为一些常见的集群信息校验失败报错案例，若未找到对应案例可根据实际的报错信息进行定位排查。

## IP Family校验不一致 （EI0001）

### 问题现象

在CANN日志中存在关键字"rank\[\*\] device ip family\[2\] is not same as others\[\*\]."，如下所示：

```text
[ERROR] HCCL(144905,python):2025-04-20-00:26:54.435.048 [config.cc:413] [145735][InitGroupStage][RanktableCheck]rank[0] device ip family[2] is not same as others[10].
```

### 可能原因

两个rank获取到的IP Family不同，比如一边是IPv4，而另一边是IPv6。

### 解决方法

查询是否配置了IPv4：

```bash
hccn_tool -i {deviceId} -ip -g
```

查询是否配置了IPv6：

```bash
hccn_tool -i {deviceId} -ip -inet6 -g
```

同一次作业的所有rank的IP Family应保持一致。HCCL默认先使用IPv4协议，若Device侧没有配置IPv4协议的IP，则会使用IPv6协议对应的ip。可以使用HCCL\_SOCKET\_FAMILY环境变量指定需要使用的网卡IP协议。

**注意**：family打印为枚举值，枚举值及对应关系如下表所示。

| IP Family枚举值 | IP协议 |
| --- | --- |
| 2 | IPv4 |
| 10 | IPv6 |

## TLS信息配置不一致（EI0016）

### 问题现象

在CANN日志中存在关键字"All ranks are consistent."，如下所示：

```text
[ERROR] HCCL(94774,all_reduce_test):2025-10-27-11:51:32.570.490 [topoinfo_exchange_agent.cc:831] [94774][InitGroupStage][RanktableCheck] Value Disable for config "tls" is invalid. Expected Value:"All ranks are consistent. Current status : rankList for enabled tls:[10.78.106.107/0]; rankList for disabled tls:[10.78.106.107/0]; rankList for query failure tls:".;
```

### 可能原因

通信域创建过程中server节点收到通信域内所有rank的信息后，会校验通信域内所有rank的tls配置是否一致，若存在配置不一致场景，则会直接校验失败退出，同时会打印出Disable或者Enable的节点列表，而未打印的节点列表则为相反的tls配置。

此校验功能仅支持在Ascend HDK 25.2.0以上的版本及通过root信息协商初始化通信域的场景中使用。Ascend 950PR/Ascend 950DT不支持此功能。

### 解决方法

1. 查询集合通信的各服务器TLS状态开关。

    在服务器中执行如下命令，获取TLS开关状态。

    ```bash
    hccn_tool -i <device_id> -tls -g
    ```

    其中<device\_id\>为Device设备的逻辑ID，您也可以通过如下for语句，一次性查询所有Device设备的TLS信息。

    ```bash
    for i in `seq 0 7`; do hccn_tool -i $i -tls -g; done    # 0，7分别为需要查询的Device ID的起始与结束值。
    ```

    打印信息如下所示：

    ```text
    dev_id:0, tls switch[0](0:disable, 1:enable), tls alarm time threshold[60]days
    dev_id:0, [pub cert] info:
             issuer[/C=CN/ST=GD/O=HUAWEI/OU=2012/CN=2_1thCA]
             start_time[Wed Feb 19 03:19:21 2020 GMT]
             end_time[Sat Feb 16 03:19:21 2030 GMT]
    dev_id:0, [ca1 cert] info:
             issuer[/C=CN/ST=GD/L=SZ/O=HUAWEI/CN=1thCA]
             start_time[Wed Feb 19 03:19:07 2020 GMT]
             end_time[Sat Feb 16 03:19:07 2030 GMT]
    dev_id:0, [ca2 cert] info:
             issuer[/C=CN/ST=GD/L=SZ/O=HUAWEI/CN=1thCA]
             start_time[Wed Feb 19 03:19:10 2020 GMT]
             end_time[Sat Feb 16 03:19:10 2030 GMT]
    dev_id:1, tls switch[0](0:disable, 1:enable), tls alarm time threshold[60]days
    dev_id:1, [pub cert] info:
             issuer[/C=CN/ST=GD/O=HUAWEI/OU=2012/CN=2_1thCA]
             start_time[Wed Feb 19 03:19:21 2020 GMT]
             end_time[Sat Feb 16 03:19:21 2030 GMT]
    dev_id:1, [ca1 cert] info:
             issuer[/C=CN/ST=GD/L=SZ/O=HUAWEI/CN=1thCA]
             start_time[Wed Feb 19 03:19:07 2020 GMT]
             end_time[Sat Feb 16 03:19:07 2030 GMT]
    dev_id:1, [ca2 cert] info:
             issuer[/C=CN/ST=GD/L=SZ/O=HUAWEI/CN=1thCA]
             start_time[Wed Feb 19 03:19:10 2020 GMT]
             end_time[Sat Feb 16 03:19:10 2030 GMT]
    ... ...
    ```

    其中tls switch\[0\]代表TLS状态为关闭，switch\[1\]代表TLS状态为开启。

2. 判断各服务器中所有Device的TLS状态开关是否一致。
    - 若不一致，建议统一修改TLS状态为开启。若TLS开关关闭，集合通信时会存在信息被窃听、篡改、仿冒的风险。

        您可以通过如下命令修改TLS状态开关：

        ```bash
        hccn_tool -i <device_id> -tls -s enable 1
        ```

        enable为状态开关，配置为1代表开启，配置为0代表关闭。

    - 若一致且状态为开启，建议您继续执行步骤3判断各节点的TLS证书信息是否一致。

3. 查看所有服务器中各Device的TLS证书信息是否一致。

    您可以通过步骤1中的信息判断各Device TLS证书信息是否一致。若不一致，您可以通过如下命令替换证书套件。

    ```bash
    hccn_tool -i 0 -tls -s path /root pri pri.pem pub pub.pem ca1 ca1.pem ca2 ca2.pem crl xxx.crl
    ```

    -i为Device ID，-path为指定证书/私钥/吊销列表存放路径，pri为私钥名字，pub为设备证书文件名，ca1/ca2/crl分别为根证书、二级根证书、吊销列表文件名。

    关于hccn\_tool工具的更多用法及参数解释，可查看对应版本的《[HCCN Tool 接口参考](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743?category=developer-documents&subcategory=interface-reference)》。

## superDeviceId重复（EI0014）

### 问题现象

在CANN日志中存在关键字"superDeviceId\[\*\*\*\] in superPod\[\*\*\*\]is already exist"，如下所示：

```text
[ERROR] HCCL(169030,alltoall_test):2025-10-23-16:28:59.392.635 [topoinfo_exchange_agent.cc:695] [169030][InitGroupStage][RanktableCheck]devices have same superDeviceId[0x3000000] in superPod[super_pod_id_0]. Current device info: serverId[127.10.0.1], rankId[0], group[hccl_world_group]. Another device info: rankId[1].
```

### 可能原因

superDeviceId是Atlas A3 训练系列产品/Atlas A3 推理系列产品内Device在超节点系统中的物理ID，是超节点系统中Device的唯一标识。HCCL在一致性校验时发现一个超节点内有相同的superDeviceId，因此校验失败。superDeviceId可通过npu-smi命令查询：

```bash
npu-smi info -t spod-info -i id -c chip_id
```

- id：设备id，通过npu-smi info -l命令查出的NPU ID即为设备id。
- chip\_id：芯片id，通过npu-smi info -m命令查出的Chip ID即为芯片id。

回显中的“SDID”即为superDeviceID。

出现此问题的可能原因是：

- 硬件配置异常。
- 通过[HCCL\_LOGIC\_SUPERPOD\_ID](../hccl_env/HCCL_LOGIC_SUPERPOD_ID.md)环境变量将不同的物理超节点配置在了同一个逻辑超节点内，导致superDeviceId重复。

### 解决方法

修改硬件配置或正确配置[HCCL\_LOGIC\_SUPERPOD\_ID](../hccl_env/HCCL_LOGIC_SUPERPOD_ID.md)环境变量，避免同一个超节点内出现superDeviceId相同的设备。
