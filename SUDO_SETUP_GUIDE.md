# Sudo Setup Guide for Jetson Benchmarking

This guide explains how to configure your system for passwordless execution of benchmark commands that require elevated privileges.

---

## Why Sudo is Required

Two commands in this project require root privileges:

1. **`make system-specs`** - Reads system telemetry and hardware specifications
2. **`make full-analysis`** - Switches power modes via `nvpmodel` command

These operations require root access because they:
- Modify system power management settings (`nvpmodel`)
- Access protected hardware telemetry interfaces (`tegrastats`)
- Read kernel-level performance counters

---

## Current Behavior (After Makefile Update)

**Improved User Experience:**
- You will be prompted for your password **once** at the start of each command
- The password prompt happens immediately (via `sudo -v`)
- Sudo credentials are cached for the duration of the command execution
- Clear messaging explains why sudo is needed

**Example:**
```bash
$ make full-analysis
Running 3-Power Mode Analysis...
Note: This command requires sudo for power mode switching
You will be prompted for your password once
[sudo] password for <your-username>: ****
[Benchmarks run without further password prompts]
```

---

## Option 1: Passwordless Sudo (Recommended for Automation)

If you want to run these commands **without any password prompts** (ideal for CI/CD, automation, or frequent benchmarking), configure passwordless sudo for specific commands.

### Setup Instructions

**Step 1: Edit sudoers file safely**
```bash
sudo visudo
```

**Step 2: Add these lines at the end** (replace `<username>` with your actual username and adjust the project path):
```bash
# Jetson benchmarking passwordless sudo
<username> ALL=(ALL) NOPASSWD: /usr/sbin/nvpmodel
<username> ALL=(ALL) NOPASSWD: /usr/bin/tegrastats
<username> ALL=(ALL) NOPASSWD: /path/to/jetson-orin-matmul-analysis/scripts/collect_system_specs.sh
```

**Step 3: Save and exit** (Ctrl+X, then Y, then Enter in nano; `:wq` in vim)

**Step 4: Test the configuration**
```bash
# Should not prompt for password
sudo nvpmodel -q

# Should not prompt for password
make system-specs
make full-analysis
```

### Security Considerations

**Secure:** Only specific commands are whitelisted
**Auditable:** All sudo usage is logged in `/var/log/auth.log`
**Limited Scope:** Privileges restricted to power management only
**Standard Practice:** Common approach for embedded system benchmarking

**Do NOT use:** `<username> ALL=(ALL) NOPASSWD: ALL` (too broad, security risk)

---

## Option 2: User Group Permissions (Partial Solution)

Add your user to system groups that provide hardware access. **Note:** This alone won't eliminate sudo requirements on Jetson, but it's good practice.

```bash
# Add user to hardware access groups
sudo usermod -aG video,i2c,gpio $USER

# Log out and back in for group changes to take effect
# Or run: newgrp video
```

**Limitations:**
- `nvpmodel` still requires sudo (NVIDIA restriction)
- `tegrastats` still requires sudo on most Jetson systems
- May help with future tools that respect group permissions

---

## Option 3: Sudo Timeout Extension

Extend the sudo password cache timeout so you don't have to re-enter your password frequently during development.

**Step 1: Edit sudoers**
```bash
sudo visudo
```

**Step 2: Add this line at the top**
```bash
Defaults timestamp_timeout=60
```

This caches your sudo password for **60 minutes** instead of the default 15 minutes.

**Trade-offs:**
- Less password prompting during active development
- Security risk if you leave your terminal unattended
- Still requires password at least once per hour

---

## Option 4: Run as Root (Not Recommended)

You could run the entire benchmarking session as root, but **this is strongly discouraged**.

```bash
# NOT RECOMMENDED
sudo su
make full-analysis
exit
```

**Why this is bad:**
- Creates files owned by root (breaks future non-root usage)
- Unnecessary security risk
- Can corrupt your virtual environment
- Against Linux best practices

---

## Recommended Configuration for Different Use Cases

### For Personal Development Workstation
Use **Option 1 (Passwordless Sudo)** + **Option 2 (User Groups)**

**Why:** Convenience for frequent benchmarking, still secure since only specific commands are whitelisted.

### For Shared Lab Equipment
Use **Option 3 (Extended Timeout)** only

**Why:** Multiple users shouldn't have passwordless sudo; extended timeout balances security and usability.

### For CI/CD Automation
Use **Option 1 (Passwordless Sudo)** exclusively

**Why:** Automation requires passwordless execution; this is the standard approach for embedded CI/CD.

### For Production Systems
**Don't run benchmarks on production hardware**

**Why:** Power mode switching can disrupt running services; benchmarks should run on dedicated development hardware.

---

## Verification

After configuring passwordless sudo (Option 1), verify with:

```bash
# Test nvpmodel (should not prompt)
sudo nvpmodel -q

# Test system specs (should not prompt)
make system-specs

# Test full analysis (should not prompt)
make full-analysis
```

If you still see password prompts, check:
1. Username in sudoers file matches your current user (`whoami`)
2. Paths to commands are absolute and correct (`which nvpmodel`)
3. No syntax errors in sudoers file (`sudo visudo -c`)

---

## Troubleshooting

### "sudo: no tty present and no askpass program specified"

**Problem:** Sudo can't prompt for a password in a non-interactive context.

**Solution:**
- Implement Option 1 (passwordless sudo)
- Or ensure you're running in an interactive terminal

### "Sorry, user <username> is not allowed to execute..."

**Problem:** Sudoers configuration is incorrect.

**Solution:**
1. Run `sudo visudo -c` to check for syntax errors
2. Verify the username matches: `whoami`
3. Ensure paths are absolute: `which nvpmodel`

### "Password required despite passwordless sudo configuration"

**Problem:** Sudoers entry doesn't match the command being executed.

**Solution:**
1. Check exact command path: `which nvpmodel`  `/usr/sbin/nvpmodel`
2. Ensure sudoers entry uses absolute paths
3. For scripts, ensure the script path is exact (no symlinks)

### "Permission denied when accessing /sys/..."

**Problem:** Even with sudo, some sysfs nodes may be read-only.

**Solution:**
- This is expected for certain GPU frequency nodes
- The benchmark tools handle these gracefully with fallback paths
- Check `data/logs/jetson_benchmark.log` for details

---

## Security Best Practices

1. **Use absolute paths** in sudoers entries
2. **Whitelist specific commands** only (never use `ALL`)
3. **Audit sudo usage** regularly: `sudo journalctl -u sudo`
4. **Keep scripts immutable**: `chmod 555 scripts/collect_system_specs.sh`
5. **Log all benchmark runs** (already done in `data/logs/`)
6. **Never commit sudoers files** to git
7. **Never share sudo passwords** with CI/CD systems
8. **Never run entire sessions as root**

---

## Documentation References

- **Jetson Power Management:** [NVIDIA Jetson Linux Developer Guide](https://docs.nvidia.com/jetson/archives/r36.3/DeveloperGuide/SD/PlatformPowerAndPerformance.html)
- **nvpmodel Documentation:** See `/etc/nvpmodel.conf` on your Jetson
- **sudoers Manual:** `man sudoers`
- **Security Considerations:** [Ubuntu Sudoers Guide](https://help.ubuntu.com/community/Sudoers)

---

## Quick Reference

| Scenario | Recommended Approach | Password Prompts |
|----------|---------------------|------------------|
| Development (personal) | Option 1: Passwordless sudo | None |
| Development (shared) | Option 3: Extended timeout | Once per hour |
| CI/CD Automation | Option 1: Passwordless sudo | None |
| Quick Testing | Current Makefile (no changes) | Once per command |
| Production Systems | Don't run benchmarks | N/A |

---

## Support

If you encounter issues with sudo configuration:

1. Check `/var/log/auth.log` for sudo errors
2. Review `data/logs/jetson_benchmark.log` for benchmark-specific issues
3. Test commands individually: `sudo nvpmodel -q`, `sudo tegrastats --interval 1000 --logfile /dev/null`
4. Verify Jetson platform: `cat /etc/nv_tegra_release`

For project-specific issues, see `CONTRIBUTING.md` or `README.md`.

---

**Last Updated:** 2025-10-13
**Project Version:** 1.0.0
**Jetson Support:** Orin Nano, Orin NX, AGX Orin (JetPack 6.x, L4T R36.x)
