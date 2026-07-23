# Reward Function

The reward is the number of packets successfully delivered in one time slot. This makes the learning objective easy to interpret: higher reward means better communication performance.

Different actions earn reward in different conditions. Active transmission works when the jammer is idle, while backscatter and rate adaptation are useful when the jammer is active. Energy harvesting gives no immediate reward but supports future transmission.

This reward design encourages the agent to balance short-term delivery with long-term adaptability under jamming.
