# Security Statement

## Recommended Running Users

To ensure security and minimize permissions, you are not advised to use administrator accounts such as `root` to execute any commands.

## File Permission Control

- You are advised to set the system `umask` value to `0027` or higher on hosts (including host machines) and containers. This ensures that new folders have a default maximum permission of `750` and new files have a default maximum permission of `640`.
- You are advised to take security measures such as permission control on sensitive files, including personal privacy data, business assets, and source files. For example, permissions for the project installation directory and public input data files must follow the recommendations in [A–Recommended Maximum Permissions for Files and Folders in Different Scenarios](#a-recommended-maximum-permissions-for-files-and-folders-in-different-scenarios).
- During installation and usage, you must enforce proper permission control, referring to the same [A–Recommended Maximum Permissions for Files and Folders in Different Scenarios](#a-recommended-maximum-permissions-for-files-and-folders-in-different-scenarios).

## Build Security Statement

When you are building and installing this project from the source code, some intermediate files will be generated. After the build is complete, you are advised to perform permission control on the intermediate files to ensure file security.

## Runtime Security Statement

If an exception occurs during running, the process exits and error information is printed. You are advised to locate the error cause based on the error information.

## Public Network Address Statement

The public network addresses contained in the code of this project are as follows.

| Type | Open-Source Code Address | File Name | Public IP Address or Public URL or Domain Name or Email or Archive File Address | Description |
| :--: | :----------: | :------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------ | :-------------------------------------------- |
| Dependency | N/A | cmake/third_party/makeself-fetch.cmake | https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz | Download the makeself source code from GitCode as a compilation dependency |
| Dependency | N/A | cmake/third_party/gtest.cmake | https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz | Download the Google Test source code from GitCode as a compilation dependency |

---

## Port Declaration

For details about the ports opened by HCCL, including transport layer protocols, authentication modes, and port usage, see the `HCCL` sheet in [CANN Communication Matrix](https://hiascend.com/document/redirect/CannCommunityCommMatrix).

## Vulnerability Handling Mechanism

[Vulnerability Management](https://gitcode.com/cann/community/blob/master/security/security.md)

## Appendix

### A-Recommended Maximum Permissions for Files and Folders in Different Scenarios

| Type | Maximum Linux Permission Reference |
| ---------------------------------- | -------------------- |
| User home directory | 750 (rwxr-x---) |
| Program files (including scripts, library files, and so on) | 550 (r-xr-x---) |
| Program file directory | 550 (r-xr-x---) |
| Configuration files | 640 (rw-r-----) |
| Configuration file directory | 750 (rwxr-x---) |
| Log files (after completion or archiving) | 440 (r--r-----) |
| Log files (during recording) | 640 (rw-r-----) |
| Log file directory | 750 (rwxr-x---) |
| Debug files | 640 (rw-r-----) |
| Debug file directory | 750 (rwxr-x---) |
| Temporary file directory | 750 (rwxr-x---) |
| Maintenance and upgrade file directory | 770 (rwxrwx---) |
| Business data files | 640 (rw-r-----) |
| Business data file directory | 750 (rwxr-x---) |
| Key component, private key, certificate, and encrypted file directory | 700 (rwx------) |
| Key component, private key, certificate, and encrypted data | 600 (rw-------) |
| Encryption and decryption interfaces and scripts | 500 (r-x------) |
