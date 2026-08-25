# TLS信息配置不一致（EI0016）

## 问题现象

在CANN日志中存在关键字"All ranks are consistent."，如下所示：

```text
[ERROR] HCCL(94774,all_reduce_test):2025-10-27-11:51:32.570.490 [topoinfo_exchange_agent.cc:831] [94774][InitGroupStage][RanktableCheck] Value Disable for config "tls" is invalid. Expected Value:"All ranks are consistent. Current status : rankList for enabled tls:[10.78.106.107/0]; rankList for disabled tls:[10.78.106.107/0]; rankList for query failure tls:".;
```

## 可能原因

通信域创建过程中server节点收到通信域内所有rank的信息后，会校验通信域内所有rank的tls配置是否一致，若存在配置不一致场景，则会直接校验失败退出，同时会打印出Disable或者Enable的节点列表，而未打印的节点列表则为相反的tls配置。

此校验功能仅支持在Ascend HDK 25.2.0以上的版本及通过root信息协商初始化通信域的场景中使用。
<!-- npu="950" id1 -->
Ascend 950PR/Ascend 950DT不支持此功能。
<!-- end id1 -->

## 解决方法

1. 查询集合通信的各服务器TLS状态开关。

    在服务器中执行如下命令，获取TLS开关状态。

    ```bash
    hccn_tool -i <device_id> -tls -g
    ```

    其中<device_id\>为Device设备的逻辑ID，您也可以通过如下for语句，一次性查询所有Device设备的TLS信息。

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

    -i为Device ID，-s path为指定证书/私钥/吊销列表存放路径，pri为私钥名字，pub为设备证书文件名，ca1/ca2/crl分别为根证书、二级根证书、吊销列表文件名。

    关于hccn_tool工具的更多用法及参数解释，可查看对应版本的《[HCCN Tool 接口参考](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743?category=developer-documents&subcategory=interface-reference)》。
