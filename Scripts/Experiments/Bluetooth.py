#!/usr/bin/python
from bluetooth import *

print("looking for bluetooth devices...")
devices = discover_devices(lookup_names=True)

for addr, name in devices:
    print("  %s - %s" % (addr, name))
print("done")