try:
    from scapy.all import sniff
    from scapy.layers.inet import IP, TCP, UDP, Ether
    from time import sleep
    import json
except ImportError:
    print("Error: Missing required libraries. Please install them using 'pip install -r requirements.txt'")
    exit(1)

class PacketSniffer:
    def __init__(self):
        print("Packet sniffing is starting...")
        sniff(prn=self.process_packet, count=1, store=False)
        

    def process_packet(self, pkt):
        print(self.extract_features_from_packet(pkt=pkt))
    
    @staticmethod
    def extract_features_from_packet(pkt):

        features = {}
        
        if pkt.haslayer(Ether):
            features['eth_dst'] = pkt[Ether].dst
            features['eth_src'] = pkt[Ether].src
            features['eth_type'] = pkt[Ether].type

        if pkt.haslayer(IP):
            features['ip_version'] = pkt[IP].version
            features['ip_ihl'] = pkt[IP].ihl
            features['ip_tos'] = pkt[IP].tos
            features['ip_len'] = pkt[IP].len
            features['ip_id'] = pkt[IP].id
            features['ip_flags'] = str(pkt[IP].flags) # R, DF, MF
            features['ip_frag'] = pkt[IP].frag
            features['ip_ttl'] = pkt[IP].ttl
            features['ip_proto'] = pkt[IP].proto
            features['ip_chksum'] = pkt[IP].chksum
            features['ip_src'] = pkt[IP].src
            features['ip_dst'] = pkt[IP].dst

        if pkt.haslayer(TCP):
            features['tcp_sport'] = pkt[TCP].sport
            features['tcp_dport'] = pkt[TCP].dport
            features['tcp_seq'] = pkt[TCP].seq
            features['tcp_ack'] = pkt[TCP].ack
            features['tcp_dataofs'] = pkt[TCP].dataofs
            features['tcp_reserved'] = pkt[TCP].reserved
            features['tcp_flags'] = str(pkt[TCP].flags) # URG, ACK, PSH, RST, SYN, FIN
            features['tcp_window'] = pkt[TCP].window
            features['tcp_chksum'] = pkt[TCP].chksum
            features['tcp_urgptr'] = pkt[TCP].urgptr

        return features 
    

if __name__ == "__main__": 
    PacketSniffer()