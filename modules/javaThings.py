import javabridge as jv
import bioformats


#############################
# javabridge configurations

def init_logger():
    """
    This is so that Javabridge doesn't spill out a lot of DEBUG messages during runtime. From CellProfiler/python-bioformats.
    """
    rootLoggerName = jv.get_static_field("org/slf4j/Logger",
                                         "ROOT_LOGGER_NAME",
                                         "Ljava/lang/String;")

    rootLogger = jv.static_call("org/slf4j/LoggerFactory",
                                "getLogger",
                                "(Ljava/lang/String;)Lorg/slf4j/Logger;",
                                rootLoggerName)

    logLevel = jv.get_static_field("ch/qos/logback/classic/Level",
                                   "WARN",
                                   "Lch/qos/logback/classic/Level;")

    jv.call(rootLogger,
            "setLevel",
            "(Lch/qos/logback/classic/Level;)V",
            logLevel)

    

def start_jvm(max_heap_size='4G'):
    """
    Start the Java Virtual Machine, enabling BioFormats IO.
    Optional: Specify the path to the bioformats_package.jar to your needs by calling.
    set_bfpath before staring to read the image data
    Parameters
    ----------
    max_heap_size : string, optional
    The maximum memory usage by the virtual machine. Valid strings
    include '256M', '64k', and '2G'. Expect to need a lot.
    """

    jv.start_vm(class_path=bioformats.JARS, max_heap_size=max_heap_size)
    init_logger()
    VM_STARTED = True

    
    
def kill_jvm():
    """
    Kill the JVM. Once killed, it cannot be restarted.
    See the python-javabridge documentation for more information.
    """
    jv.kill_vm()
    VM_KILLED = True    
