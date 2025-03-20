"""
Person tracking and state management for entry/exit detection
"""

import csv
from datetime import datetime

class PersonTracker:
    def __init__(self):
        self.person_states = {}  # {person_id: [identity, last_state, last_seen, confidence]}
        self.log_file = 'person_tracking.csv'
        self.initialize_log()
        
    def initialize_log(self):
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Person', 'Action', 'Confidence'])
    
    def update_state(self, person_id, identity, current_state, confidence):
        timestamp = datetime.now()
        
        # Get previous state
        prev_state = None
        if person_id in self.person_states:
            prev_state = self.person_states[person_id][1]
        
        # Update state if changed
        if prev_state != current_state:
            action = "Entered" if current_state == 0 else "Exited"
            self.log_activity(timestamp, identity, action, confidence)
        
        # Update tracking
        self.person_states[person_id] = [identity, current_state, timestamp, confidence]
    
    def log_activity(self, timestamp, identity, action, confidence):
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, identity, action, confidence])
    
    def handle_disappearance(self, current_persons):
        """Handle persons who disappeared from frame"""
        timestamp = datetime.now()
        disappeared = set(self.person_states.keys()) - set(current_persons)
        
        for person_id in disappeared:
            if (timestamp - self.person_states[person_id][2]).seconds > 5:
                # Person truly disappeared, log final state
                identity = self.person_states[person_id][0]
                last_state = "Inside" if self.person_states[person_id][1] == 0 else "Outside"
                self.log_activity(timestamp, identity, f"Lost tracking ({last_state})", 
                                self.person_states[person_id][3])
                del self.person_states[person_id]
