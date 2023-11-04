from tminterface.client import Client
from tminterface.interface import TMInterface
from tminterface.constants import DEFAULT_SERVER_SIZE
import signal
import time
import configparser



class MapLoader(Client):
    def __init__(self, map):
        super(MapLoader,self).__init__()
        self.map=map
        self.finished=False

    def on_registered(self, iface: TMInterface):
        """ A callback that the client has registered to a TMInterface instance.
        """
        print(f'Registered to {iface.server_name}')
        #iface.log("Ready. Genome id: " + str(self.genome_id))
        #set gamespeed
        iface.set_timeout(5000)
        iface.execute_command("map "+self.map)
        iface.give_up()
        iface.close()
        self.finished=True
    

    def on_deregistered(self,iface):
        """ A callback that the client has been deregistered from a TMInterface instance. 
        This can be emitted when the game closes, the client does not respond in the timeout window, 
        or the user manually deregisters the client with the deregister command.
        """
        print(f'deregistered to {iface.server_name}')

def LoadMap(map):
    client=MapLoader(map)
    buffer_size=DEFAULT_SERVER_SIZE
    server_name="TMInterface0"
    iface = TMInterface(server_name, buffer_size)

    def handler(signum, frame):
        iface.close()

    # Close connections
    signal.signal(signal.SIGBREAK, handler)
    signal.signal(signal.SIGINT, handler)

    # Register a new client
    iface.register(client)
    while not client.finished:
        time.sleep(0)
    iface.close()
    time.sleep(5)


if __name__=="__main__":
    run_config="../models/config.ini"
    config_file = configparser.ConfigParser()
    config_file.read(run_config)

    map = config_file['Map']['straight']
    print(map)
    LoadMap(map)