=======
Network
=======

.. meta::
   :description: Linux network commands covering configuration, diagnostics, port scanning, packet capture, DNS lookup, data transfer, and firewall management.
   :keywords: Linux networking, nmap, tcpdump, netstat, curl, wget, iptables, network diagnostics, port scanning, packet capture

.. contents:: Table of Contents
    :backlinks: none

Network troubleshooting demands a diverse toolkit. Whether you're diagnosing connectivity
issues with ping and traceroute, scanning ports with nmap, capturing packets with tcpdump,
or configuring firewall rules with iptables, these commands form the foundation of network
administration and security auditing.

An interactive demo script is available at `src/bash/net.sh <https://github.com/crazyguitar/cppcheatsheet/blob/master/src/bash/net.sh>`_
to help you experiment with the concepts covered in this cheatsheet.

.. code-block:: bash

    ./src/bash/net.sh           # Run all demos
    ./src/bash/net.sh dns       # Run DNS lookup demo
    ./src/bash/net.sh curl      # Run curl examples
    ./src/bash/net.sh --help    # Show all available sections

Network Configuration
=====================

Commands for viewing and configuring network interfaces and routing.

Interface Information
---------------------

.. code-block:: bash

    ip addr                             # IP addresses
    ip addr show eth0                   # Specific interface
    ip -4 addr                          # IPv4 only
    ip -6 addr                          # IPv6 only
    ip link                             # Link layer info
    ip link show eth0                   # Interface status

    ifconfig                            # Legacy interface info
    ifconfig eth0                       # Specific interface

Interface Management
--------------------

.. code-block:: bash

    ip link set eth0 up                 # Bring interface up
    ip link set eth0 down               # Bring interface down
    ip addr add 192.168.1.10/24 dev eth0    # Add IP address
    ip addr del 192.168.1.10/24 dev eth0    # Remove IP address

Routing
-------

.. code-block:: bash

    ip route                            # Routing table
    ip route show                       # Same as above
    ip route get 8.8.8.8                # Route to destination
    ip route add default via 192.168.1.1    # Add default gateway
    ip route add 10.0.0.0/8 via 192.168.1.1 # Add static route
    ip route del 10.0.0.0/8             # Delete route

    route -n                            # Legacy routing table
    netstat -rn                         # Routing table

ARP Table
---------

.. code-block:: bash

    ip neigh                            # ARP table
    ip neigh show                       # Same as above
    arp -a                              # Legacy ARP table
    ip neigh flush all                  # Clear ARP cache

Connection Information
======================

Commands for viewing active connections and listening ports.

Socket Statistics
-----------------

.. code-block:: bash

    ss -tuln                            # TCP/UDP listening ports
    ss -tulnp                           # Include process info
    ss -t                               # TCP connections
    ss -u                               # UDP connections
    ss -x                               # Unix sockets
    ss -s                               # Socket summary
    ss -o state established             # Established connections

    # Filter by port
    ss -tuln | grep :80
    ss -tuln sport = :443

    # Filter by state
    ss -t state time-wait
    ss -t state established

Netstat
-------

.. code-block:: bash

    # Linux
    netstat -tuln                       # Listening ports
    netstat -tulnp                      # With process info
    netstat -an                         # All connections
    netstat -s                          # Protocol statistics
    netstat -i                          # Interface statistics

    # macOS/BSD
    netstat -an                         # All connections
    netstat -p tcp                      # TCP connections
    netstat -p udp                      # UDP connections
    netstat -rn                         # Routing table
    netstat -an | grep LISTEN           # Listening ports

Process and Ports
-----------------

.. code-block:: bash

    lsof -i :80                         # Process using port 80
    lsof -i :8080 -i :443               # Multiple ports
    lsof -i tcp                         # All TCP connections
    lsof -i @192.168.1.1                # Connections to host
    fuser 80/tcp                        # PID using port

Connectivity Testing
====================

Commands for testing network connectivity and diagnosing issues.

Ping
----

.. code-block:: bash

    ping host                           # Continuous ping
    ping -c 4 host                      # 4 packets
    ping -i 0.2 host                    # 200ms interval
    ping -s 1000 host                   # Packet size 1000 bytes
    ping -W 2 host                      # 2 second timeout
    ping -q -c 10 host                  # Quiet, summary only

Traceroute
----------

.. code-block:: bash

    traceroute host                     # Trace route
    traceroute -n host                  # No DNS resolution
    traceroute -T host                  # TCP instead of UDP
    traceroute -p 443 host              # Specific port
    traceroute -m 15 host               # Max 15 hops

    tracepath host                      # MTU discovery
    mtr host                            # Combined ping/traceroute
    mtr -r -c 10 host                   # Report mode

Port Testing
------------

.. code-block:: bash

    # netcat (nc)
    nc -zv host 80                      # Test TCP port
    nc -zv host 20-30                   # Port range
    nc -uzv host 53                     # Test UDP port
    nc -w 3 -zv host 443                # 3 second timeout

    # telnet
    telnet host 80                      # Connect to port

    # bash built-in
    timeout 3 bash -c "</dev/tcp/host/80" && echo "open"

DNS Lookup
==========

Commands for querying DNS records and troubleshooting name resolution.

dig
---

.. code-block:: bash

    dig example.com                     # A record
    dig example.com A                   # Explicit A record
    dig example.com AAAA                # IPv6 record
    dig example.com MX                  # Mail servers
    dig example.com NS                  # Name servers
    dig example.com TXT                 # TXT records
    dig example.com ANY                 # All records

    dig +short example.com              # Short answer
    dig +trace example.com              # Trace delegation
    dig +noall +answer example.com      # Answer only

    dig @8.8.8.8 example.com            # Use specific DNS
    dig -x 8.8.8.8                      # Reverse lookup

Other DNS Tools
---------------

.. code-block:: bash

    nslookup example.com                # Name server lookup
    nslookup -type=mx example.com       # MX records
    nslookup example.com 8.8.8.8        # Use specific DNS

    host example.com                    # Simple lookup
    host -t mx example.com              # MX records
    host 8.8.8.8                        # Reverse lookup

    getent hosts example.com            # NSS lookup
    resolvectl query example.com        # systemd-resolved

Data Transfer
=============

Commands for downloading files and making HTTP requests.

curl
----

.. code-block:: bash

    curl https://example.com            # GET request
    curl -o file.html https://url       # Save to file
    curl -O https://url/file.zip        # Save with original name
    curl -I https://example.com         # Headers only
    curl -v https://example.com         # Verbose output
    curl -s https://example.com         # Silent mode
    curl -L https://example.com         # Follow redirects

    # POST requests
    curl -X POST https://url
    curl -d "data=value" https://url    # POST with data
    curl -d @file.json https://url      # POST from file
    curl -H "Content-Type: application/json" -d '{"key":"value"}' https://url

    # Headers and auth
    curl -H "Authorization: Bearer token" https://url
    curl -u user:pass https://url       # Basic auth
    curl -b "cookie=value" https://url  # Send cookie
    curl -c cookies.txt https://url     # Save cookies

    # Other options
    curl -k https://url                 # Ignore SSL errors
    curl --connect-timeout 5 https://url
    curl -x proxy:port https://url      # Use proxy

wget
----

.. code-block:: bash

    wget https://url/file               # Download file
    wget -O name.zip https://url        # Save with name
    wget -c https://url/file            # Resume download
    wget -q https://url                 # Quiet mode
    wget -b https://url                 # Background

    wget -r https://example.com/        # Recursive download
    wget -r -np https://example.com/    # No parent
    wget -m https://example.com/        # Mirror site
    wget -i urls.txt                    # Download from list

    wget --limit-rate=1m https://url    # Limit bandwidth
    wget --user=u --password=p https://url

Remote Copy
-----------

.. code-block:: bash

    scp file user@host:/path            # Copy to remote
    scp user@host:/path/file .          # Copy from remote
    scp -r dir user@host:/path          # Recursive copy
    scp -P 2222 file user@host:/path    # Custom port

    rsync -avz src/ user@host:/dest/    # Sync directories
    rsync -avz --delete src/ dest/      # Mirror with delete
    rsync -avzP src/ dest/              # Progress bar
    rsync --dry-run -avz src/ dest/     # Dry run

Port Scanning with nmap
=======================

Nmap is a powerful network scanner for security auditing and network discovery.

Basic Scanning
--------------

.. code-block:: bash

    nmap host                           # Default scan (top 1000 ports)
    nmap -p 80 host                     # Single port
    nmap -p 80,443,8080 host            # Multiple ports
    nmap -p 1-1000 host                 # Port range
    nmap -p- host                       # All 65535 ports
    nmap -F host                        # Fast scan (top 100)

    nmap 192.168.1.0/24                 # Scan subnet
    nmap 192.168.1.1-50                 # IP range
    nmap -iL hosts.txt                  # From file

Scan Types
----------

.. code-block:: bash

    nmap -sT host                       # TCP connect scan
    nmap -sS host                       # SYN scan (stealth)
    nmap -sU host                       # UDP scan
    nmap -sA host                       # ACK scan
    nmap -sN host                       # NULL scan
    nmap -sF host                       # FIN scan
    nmap -sX host                       # Xmas scan

Host Discovery
--------------

.. code-block:: bash

    nmap -sn 192.168.1.0/24             # Ping scan (no port scan)
    nmap -Pn host                       # Skip host discovery
    nmap -PS22,80,443 host              # TCP SYN discovery
    nmap -PA80,443 host                 # TCP ACK discovery
    nmap -PU53 host                     # UDP discovery
    nmap -PR 192.168.1.0/24             # ARP discovery (local)

Service and Version Detection
-----------------------------

.. code-block:: bash

    nmap -sV host                       # Service version detection
    nmap -sV --version-intensity 5 host # More aggressive
    nmap -A host                        # OS + version + scripts
    nmap -O host                        # OS detection
    nmap --osscan-guess host            # Aggressive OS guess

Nmap Scripts (NSE)
------------------

.. code-block:: bash

    nmap --script=default host          # Default scripts
    nmap --script=vuln host             # Vulnerability scripts
    nmap --script=safe host             # Safe scripts only
    nmap --script=http-headers host     # Specific script
    nmap --script=http-* host           # Wildcard scripts

    # Common scripts
    nmap --script=ssl-enum-ciphers -p 443 host
    nmap --script=http-title host
    nmap --script=ssh-hostkey host
    nmap --script=dns-brute host
    nmap --script=smb-os-discovery host

Output Formats
--------------

.. code-block:: bash

    nmap -oN scan.txt host              # Normal output
    nmap -oX scan.xml host              # XML output
    nmap -oG scan.gnmap host            # Grepable output
    nmap -oA scan host                  # All formats

Timing and Performance
----------------------

.. code-block:: bash

    nmap -T0 host                       # Paranoid (slowest)
    nmap -T1 host                       # Sneaky
    nmap -T2 host                       # Polite
    nmap -T3 host                       # Normal (default)
    nmap -T4 host                       # Aggressive
    nmap -T5 host                       # Insane (fastest)

    nmap --min-rate 1000 host           # Min packets/sec
    nmap --max-retries 2 host           # Max retries

Packet Capture with tcpdump
===========================

Tcpdump captures and analyzes network traffic for debugging and security analysis.

Basic Capture
-------------

.. code-block:: bash

    tcpdump                             # Capture all traffic
    tcpdump -i eth0                     # Specific interface
    tcpdump -i any                      # All interfaces
    tcpdump -c 100                      # Capture 100 packets
    tcpdump -w capture.pcap             # Write to file
    tcpdump -r capture.pcap             # Read from file

Filtering
---------

.. code-block:: bash

    # By host
    tcpdump host 192.168.1.1
    tcpdump src 192.168.1.1
    tcpdump dst 192.168.1.1

    # By port
    tcpdump port 80
    tcpdump src port 443
    tcpdump dst port 22
    tcpdump portrange 8000-9000

    # By protocol
    tcpdump tcp
    tcpdump udp
    tcpdump icmp
    tcpdump arp

    # Combine filters
    tcpdump 'host 192.168.1.1 and port 80'
    tcpdump 'src 192.168.1.1 and tcp'
    tcpdump 'port 80 or port 443'
    tcpdump 'not port 22'

Output Options
--------------

.. code-block:: bash

    tcpdump -n                          # No DNS resolution
    tcpdump -nn                         # No DNS or port names
    tcpdump -v                          # Verbose
    tcpdump -vv                         # More verbose
    tcpdump -X                          # Hex and ASCII
    tcpdump -A                          # ASCII only
    tcpdump -e                          # Show MAC addresses
    tcpdump -tttt                       # Human-readable time

Advanced Filters
----------------

.. code-block:: bash

    # TCP flags
    tcpdump 'tcp[tcpflags] & tcp-syn != 0'
    tcpdump 'tcp[tcpflags] & tcp-rst != 0'
    tcpdump 'tcp[13] & 2 != 0'          # SYN packets

    # HTTP traffic
    tcpdump -A 'tcp port 80 and (((ip[2:2] - ((ip[0]&0xf)<<2)) - ((tcp[12]&0xf0)>>2)) != 0)'

    # Packet size
    tcpdump 'greater 1000'              # Packets > 1000 bytes
    tcpdump 'less 100'                  # Packets < 100 bytes

Firewall (iptables)
===================

Commands for managing Linux firewall rules with iptables.

View Rules
----------

.. code-block:: bash

    iptables -L                         # List rules
    iptables -L -n                      # Numeric output
    iptables -L -v                      # Verbose
    iptables -L -n -v --line-numbers    # With line numbers
    iptables -S                         # Rules as commands
    iptables -t nat -L                  # NAT table

Basic Rules
-----------

.. code-block:: bash

    # Allow incoming SSH
    iptables -A INPUT -p tcp --dport 22 -j ACCEPT

    # Allow incoming HTTP/HTTPS
    iptables -A INPUT -p tcp --dport 80 -j ACCEPT
    iptables -A INPUT -p tcp --dport 443 -j ACCEPT

    # Allow established connections
    iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

    # Drop all other incoming
    iptables -A INPUT -j DROP

    # Allow from specific IP
    iptables -A INPUT -s 192.168.1.100 -j ACCEPT

    # Block specific IP
    iptables -A INPUT -s 10.0.0.5 -j DROP

Rule Management
---------------

.. code-block:: bash

    # Delete rule by number
    iptables -D INPUT 3

    # Delete rule by specification
    iptables -D INPUT -p tcp --dport 80 -j ACCEPT

    # Insert rule at position
    iptables -I INPUT 1 -p tcp --dport 22 -j ACCEPT

    # Flush all rules
    iptables -F
    iptables -t nat -F

    # Save/restore rules
    iptables-save > rules.txt
    iptables-restore < rules.txt

Network Bandwidth
=================

Commands for monitoring and testing network bandwidth.

.. code-block:: bash

    # iftop - interface bandwidth
    iftop                               # Default interface
    iftop -i eth0                       # Specific interface
    iftop -n                            # No DNS resolution

    # nethogs - bandwidth by process
    nethogs                             # All interfaces
    nethogs eth0                        # Specific interface

    # iperf3 - bandwidth testing
    iperf3 -s                           # Server mode
    iperf3 -c server_ip                 # Client mode
    iperf3 -c server_ip -t 30           # 30 second test
    iperf3 -c server_ip -R              # Reverse mode
    iperf3 -c server_ip -u              # UDP test
