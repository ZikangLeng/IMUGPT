try:
  from utils.sliding_window import sliding_window
  from utils.ecdfRep import ecdfRep
  from utils.statRep import statRep
except:
  from .sliding_window import sliding_window  
  from .ecdfRep import ecdfRep
  from .statRep import statRep