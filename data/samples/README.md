# Sample logs

`sysmon_sample.xml` is a **synthetic**, hand-written Sysmon export (13 events) used
by tests and demos. Hosts, users and domains are fictional and external IPs come
from the RFC 5737 documentation ranges. It contains no working payloads.

The sample tells one small intrusion story (macro document -> PowerShell ->
payload drop -> Run-key persistence -> LSASS access -> outbound beacon) mixed with
benign browsing, plus one command line carrying a prompt-injection string aimed
at an AI analyst.
