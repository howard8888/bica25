#!/usr/bin/env python3
# pylint: disable=line-too-long
"""
##Section 0: Preface
##------------------
#
# Software written by someone else is often hard to read and understand.
#
# To make this software as readable as possible, I am writing and organizing this code from the philosophy that computer
#   programs are written for people, not computers. I try to tell a story with any program, whether comments are short or long. I am writing
#   from a first-person point of view, rather than more formally so that you the reader can better follow and understand the code below.
#
#   (In a team/corporate environment it may not be possible to do this, but the reality is that the amount of cognitive overload of the
#    human brain in reading software is not appreciated. Coders use their IDEs to jump from one line docstrings to another, and while
#    this gives the impression of efficiency, it is not reflected in software maintenance nor good understand of the underlying code.)
#
# Level of Python knowledge required -- The level of Python knowledge required is an intermediate Python level. You do have to have
#    some fluency in Python for the code below to make sense, notwithstanding the extensive docstrings provided.
#
#   (A quick apology about type hints -- they help me write, debug and maintain code. But you can ignore all the typing, i.e.,  ignore
#    all the List[Tuple[int, int,...etc...]], unless you want to modify the code, in which case they can be useful.)


##Section 1: Introduction -- What is the Purpose of this Code?
##------------------------------------------------------------

Simulation to accompany paper: The Two-Stream Hypothesis as a Foundation for Human-like Memory and Action
Howard Schneider, May 2025
hschneidermd@alum.mit.edu

In the two-stream hypothesis, the mammalian brain’s visual (and possibly other sensory) information diverges into a
 ventral “what” pathway for object identity and a dorsal “where” pathway for spatial planning. Although this anatomical split
 is well documented, its computational value and its role in the emergence of human-like intelligence remain debated. We combine
 a neurobiological review with targeted simulations to evaluate when a split model (i.e., “what” and “where” facts are stored
 separately) outperforms a unified model (i.e., facts stored together). This code below is intended to create the targested
 simulations.

Code developed with Python 3.11.4 under Windows 11 Pro command line terminal
Run-time OS: Windows 11 Pro build 26100.3195 via command line terminal
Run-time hardware: Intel i7-1360P, 2.60 GHz, 16GB RAM, no GPU usage
Note: Results will vary somewhat depending on OS, hardware as well as stochastically from run to run
(If you have large issues, try adjusting default parameters. Regardless, they will be equally applied to Split and Unified models.)

To run this code in Windows:
-make sure you have Python 3.11 installed (or as part of a virtual environment) (if multiple versions, specify at the command line)
(if you enter >python at the command line you will see the version; then enter >>>exit() to exit out of the Python REPL)
-open the command prompt or powershell
-go to the directory where you have put this file
-make sure a valid pandas library is installed or else install it:  >pip install pandas
-enter at the command line  >python two_stream_01.py

To run this code in macOS:
-similar to above, but you may have Python 2 and Python 3 installed, so make sure a compatible version of Python is chosen

To run this code in Linux:
-again, you may have Python 2 and Python 3 installed, so make sure a compatible version of Python is chosen
(nb. Will not run in Python 2. With regard to different versions of Python 3, I have not done the testing at this point.)

To furthe develop this code:
Note: pylint, mypy and pytest are used in the development process but are not used by the actual run-time code
To install same libraries used by this code:   >pip install -r requirements.txt
requirements.txt:
pandas>=2.2.2pi
pylint>=3.2.6
mypy>=1.11.0
pytest>=8.3.5

The code of this simulation is currently all in one single Python file; no other package handling required.
The imports are intentionally all part of the standard library so no installation required except for pandas as noted above.
(I would have liked to have avoided any library installation by the end user, but too much coding without using pandas or equivalent.)
Not included in the code below:
-Unit tests
-Matplotlib plotting code that created the graphs in Figure 2
-CCA8 architecture code
"""

##Section 2 -- What Extra Code is Involved in this Program?
##---------------------------------------------------------
#
# Just look through quickly. Who is involved in this code, i.e., are there going to be strange imports that will create the bulk of the
#   functionality of this code. Have you seen these players before? Also, will there be a bunch of global variables roaming around the
#   storyline?  Pragmas -- you can ignore (unless you want to modify the code later).

## START IMPORTS
import random
import time
import argparse
import os
import multiprocessing as mp
from typing import Dict, List, Any, Tuple, Union
import pandas as pd   #type: ignore
#due to GIL (i.e., global interpreter lock) readers may be unfamiliar with multiprocessing with Python, but it is required for our simulations
#  -unlike the threading library which uses threads within a *single* process, here we use multiprocessing library to create multiple *separate* processes
#  -we will bypass Python's GIL with these separate processes and therefore simulate the parallelism the hypotheses of the paper require
#  -future versions of Python may include readily accessible mechanisms to bypass GIL, but at the time of this writing, GIL lock remains an issue for parallelism

## END IMPORTS

##START GLOBALS, CONSTANTS, INITIALIZATIONS
FEATURE_PAD: int = 128
#
# -generally try to avoid explicity globals but effectively do the same thing via classes
##END GLOBALS, CONSTANTS, INITIALIZATIONS


##START PRAGMAS
#
# nb. my personal list of pylint disables, not all may apply to this program
# pylint: disable=line-too-long
# pylint: disable=invalid-name
# pylint: disable=bare-except
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
# pylint: disable=too-many-instance-attributes
# pylint: disable=unused-wildcard-import
# pylint: disable=wildcard-import
# pylint: disable=pointless-string-statement
# pylint: disable=too-many-lines
# pylint: disable=too-many-nested-blocks
# pylint: disable=too-many-public-methods
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=multiple-statements
# pylint: disable=wrong-import-position
# pylint: disable=use-implicit-booleaness-not-comparison
# pylint: disable=too-many-return-statements
# pylint: disable=no-else-return
# pylint: disable=logging-fstring-interpolation
#
##END PRAGMA


##START FUNCTIONS, CLASSES

##Section 3 --Create data for the Simulation and classes representative of the two models being compared (see paper)
##------------------------------------------------------------------------------------------------------------------
#
# In this section we set up the background of our story. In the Introduction we talked about comparing a Split model with a
#   Unified model, in order to test our two-stream hypothesis as described in the accompanying paper.
# Well... at this point we now have to set up the characters of our story. Rather than directly set up the characters and create about
#   mess of software, we will use a Split class and a Unified class to later allow instantiation of a Split character of our story (i.e, the
#   Split model) and a Unified character of our story (i.e., Unified model). Within these classes we can encapsulate the dictionaries
#   each of these classes needs to use.
# The dictionaries of each of these classes needs to be generated from a dataset common to both models, so we will create that, i.e.,
#   in a large grid x grid array, we will randomly place n (e.g. n=5000) different objects. This will be our dataset, and then from this
#   common dataset we can create the dictionaries needed by each of the classes to hold "what" info, "where" info and/or "integrated" (both what
#   and where) info.
# Then we can create methods within the classes to operate on these dictionaries.
# Once this is all done, later on in the code to create the two main characters of our story, i.e., Split and Unified, and do various "what" and
#   "where" and "integrated" operations, we can do so via all the data and methods neatly encapsulated by these classes.
#
# If you want just enough information to understand how the code works in the sections below, just look at the concise cheatsheet.
#
# If you want to have a bit of deeper information to understand how the code works in the sections below, i.e., where are the
#   "where" and "what" information coming from, then look at the full cheatsheet below.
#
# If you want to see how the hypotheses of the paper were implemented in the Unified and the Split classes which represent the
#   Unified model and the Split model in our testing of the two-stream hypothesis, as per the paper, then look at the actual code of these
#   classes. The docstrings of the classes and their methods are well documented below.
#
#
##Concise cheatsheet of Unified and Split methods (know this before reading the next section 4 of code)
#"c" is "where", "f" is "what", "oid" is an internal pointer without semantic information
#self.oid_to_coord(oid) --> c
#self.coord_to_oid(c) --> oid
#self.full_lookup(oid)--> oid, f
#
#
##Cheatsheet of dataset and Class Unified and Split dictionaries and methods:
#this cheatsheet summarizes the dictionaries used by these classes
# gen_ds(n object, grid size, * rng seeding) --> [[ (0, (x,y), 'obj000000XXX....XXX'), ....] == {(oid==obj_id, c==(x,y), f=feature_string), ....} <--"ds"
# class Unified (ds)
# -- self.id_map = {oid: (c,f), ...} ==  {....., 4999: ((93225264, 70034202), 'obj0004999XX....XX')}
# -- self.coord_map = {c: (oid, f), ...} == {....,(93225264, 70034202): (4999, 'obj0004999XX....XX')
# ---- self.oid_to_coord(oid) --> self.id_map[oid][0] --> c ==  e.g., 4999 --> (93225264, 70034202)
# ---- self.coord_to_oid(c) --> self.coord_map[coord][0] --> oid  == e.g., (93225264, 70034202) --> 4999
# ---- self.full_lookup(oid) --> self.id_map[oid] --> (oid,f) == e.g., 4999 --> ((93225264, 70034202), 'obj0004999XX....XX')
# class Split(ds)
# -- self.where = {oid:c, ...} = {...., 4999:(93225264, 70034202)   }
# -- self.what = {oid:f, ...}  = {...., 4999:'obj0004999XX....XXX'  }
# -- self.idx = {c:oid, ...}   = {....,  (93225264, 70034202): 4999 }
# ---- self.oid_to_coord(oid) --> self.where[oid] --> c = e.g., 4999 --> (93225264, 70034202)
# ---- self.coord_to_oid(c) --> self.idx[c] --> oid  == e.g., (93225264, 70034202) --> 4999
# ---- self.full_lookup(oid) --> c=self.where[oid], f=self.what[oid] == e.g., 4999 -->(93225264, 70034202), 'obj0004999XX....XX'
#

def gen_ds(n: int, grid: int, *, rng: random.Random) -> List[Tuple[int, Tuple[int, int], str]]:
    """This function generates n unique objects, each assigned a random coordinate
    on a grid × grid space, along with a fixed-width string label
    like "obj0000003XXXXXX...".

    input parameters:
    n -- int, use in range(n) to set a range which will give total number of objects to generates
    grid -- int, defines a grid from (0,0) to (grid, grid), i.e., the coordinate space in which we will randomly be putting
          the n objects in this method (larger a space, then less likely collisions of occupied spots)
    *, rng -- forces code calling this method to name the rng argument, e.g., gen_ds(1000, 5000, rng=my_rng) is ok
         versus gen_ds(1000, 5000, my_rng) will give an error
         - note that it is defined as a keyword-only parameter (i.e., it is after the asterisk in the definition)
           (note to the reader: largely taken from random module specs; rng must be seeded RNG (random number generator) object,
           e.g., my_rng = random.Random(42) then it is ok to put rng.randint()

    returns:
    row -- list which is composed of (object id + c + text "obj00000"+object_id + pads with X's until length==128) for
            each of the object_id's counted
           for example:
                input: gen_ds(5, 4, rng=random.Random(23))
                output: rows = [ (0, (x,y), 'obj000000XXX....XXX'), (1, (x,y), 'obj000001XXX.....XXX) .....
                                                    (4, (x,y), 'obj000001XXX.....XXX)) ]
    explanation:
        -we count object_id from 0 to n-1 objects
        -the coordinate space in which we can assign random x,y=="c" coordinates of an object is limited by grid x grid parameter value
        -we assign a random x,y to c, i.e., c=(x,y) (which are seeded by *rng per the random library specifications)
        -if c is a new coordinate, we add it to the set used (if not we choose new x,y)
        -we then append to to the list rows (object_id + (x,y) + "object00001" (if object_id is 1) +pad with X's)
        -so this utility function produces our gridxgrid array of n=5000 (or other n) objects at random x,y locations
            [ tuple [oid int, tuple[random x,y]]] , .... ]
    """
    used: set[Tuple[int, int]] = set()
    rows: List[Tuple[int, Tuple[int, int], str]] = []
    for object_id in range(n):
        while True:
            c = (rng.randint(0, grid), rng.randint(0, grid))
            if c not in used:
                used.add(c)
                break
        rows.append((object_id, c, f"obj{object_id:07d}".ljust(FEATURE_PAD, "X")))
    return rows


class Unified:
    """single table memory architecture -- *what* and *where* reside together
    -this class creates the Unified model which we use to compare against the Split model
    -instantiated the class with parameter ds which is the gridgrid array of n objects produced by gen_ds()
    -instance variable self.id_map uses a dict comprehension to build {oid: c,f}
        remember -- ds is a list of 3-element tuples (created above by gen_ds) -- (oid, coord, label)
            where 'oid' is the object number, coord is the (x,y), and label is eg. 'obj000004XXXXX....XXXX'
                 -- thus, the dict comprehension is doing this eg, {  004: ((14, 33), 'obj000004XXXXX....XXXX' ), ....}
                 -- for each record (oid, coord, label) the dict comprehension builds a dictionary entry of oid: (coord, label)
                 -- this is actual last dictionary element in self.id_map
                               {..... ,4999: ((93225264, 70034202), 'obj0004999XXXXXXXXXXXXXX....XXXX')  }
    -note that self.id_map combines location and feature information in the same dictionary
    -instance variable self.coord_map builds a dictionary from ds but each element is (x,y): (oid, label)
                 -- this is actual last dictionary element in self.id_map
                             {....,(93225264, 70034202): (4999, 'obj0004999XXXXX....XXXXXXXXXXX')       }
    -self.id_map supports oid input and get the (x,y) as output, i.e., what --> where
    -self.coord_map supports (x,y) input and get the oid as output, i.e., where --> what
    -actually, this class essentially takes in ds and builds up self.id_map and self.coord_map and has
       methods which either return (x,y) for an oid, or return oid for an (x,y)
    """

    def __init__(self, ds: List[Tuple[int, Tuple[int, int], str]]):  #ds = list of n objects in the gridxgrid array produced by gen_ds() above
        self.id_map: Dict[int, Tuple[Tuple[int, int], str]] = {oid: (c, f) for oid, c, f in ds} #{..., oid:((x,y), 'objXXXX')}
        self.coord_map: Dict[Tuple[int, int], Tuple[int, str]] = {c: (oid, f) for oid, c, f in ds} #{...., (x,y): (oid, 'objXXXX')}

    def oid_to_coord(self, oid: int) -> Tuple[int, int]:
        """return the *(x, y)* coordinate of object *oid*
        -self.id_map supports oid input and get the (x,y) as output, i.e., what --> where
        -- for each record (oid, coord, label) the dict comprehension builds a dictionary entry of oid: (coord, label)
        -- this is actual last dictionary element in self.id_map
             {..... ,4999: ((93225264, 70034202), 'obj0004999XXXXXXXXXXXXXX....XXXX')  }
             self.id_map[oid][0] e.g., 4999 --> (93225264, 70034202)
        """
        return self.id_map[oid][0]

    def coord_to_oid(self, coord: Tuple[int, int]) -> int:
        """Return the *oid* found at *coord*.
        -instance variable self.coord_map builds a dictionary from ds but each element is (x,y): (oid, label)
        -- this is actual last dictionary element in self.id_map
                {....,(93225264, 70034202): (4999, 'obj0004999XXXXX....XXXXXXXXXXX')       }
        """
        return self.coord_map[coord][0]

    def full_lookup(self, oid: int) -> Tuple[Tuple[int, int], str]:
        """Return the *what*/*where* tuple for *oid* (integrated lookup).
        -self.id_map supports oid input and get the (x,y) as output, i.e., what --> where
        -- for each record (oid, coord, label) the dict comprehension builds a dictionary entry of oid: (coord, label)
        -- this is actual last dictionary element in self.id_map
             {..... ,4999: ((93225264, 70034202), 'obj0004999XXXXXXXXXXXXXX....XXXX')  }
             self.id_map[oid] e.g., 4999 --> ((93225264, 70034202), 'obj0004999XXXXXXXXXXXXXX....XXXX'))
        """
        return self.id_map[oid]


class Split:
    """two-table memory architecture --*what* and *where* reside appart
    -used to represent the ventral/dorsal split hypothesis model
    -please see the docstring for class Unifed above before reading this docstring, as the two share and contrast features based on the underlying model
    -again, this class is instantiated with ds -- the list of n objects in the gridxgrid array produced by gen_ds() above
    -above we are using one unified table (hence class "Unified") which can be stored as an id_map and a coord_map as we do above and what we call them there
    -now for Split class we are modeling the dorsal/ventral model where there is canonically what and where stores, and that is what we call them here
        (note -- original names self.where_store and self.what_store have been shorted to self.where and self.what)
    -ok... some things to have in your memory at this point as you read the code:
       -oid == object_id == the object's ID used for coding purposes (the "what" is the label e.g., "obj000372XXX...XXX"; the "where" is the random (x,y) in gridxgrid)
       -above we created self.id_map and self.coord_map -- when you see these, think 'unified model'  (although... I will try my best to keep encapsulated)
    -now we will create self.where, self.what and self.idx -- when you see these, think 'split model'  (although again... I will try my best to keep encapsulated)
    -instance variable self.where uses a dict comprehension to build {oid: (x,y)}
        -remember -- ds is a list of 3-element tuples (oid, c==(x,y), _==label) -- just step through oid & c in ds and construct the dict comprehension self.where
        -self.where = {...., 4999:(93225264, 70034202)  }  <-- last actual entry, note oid:(x,y) pairs in this dictionary
    -instance variable self.what uses a dict comprehension to build {oid: f==label}
        -remember -- ds is a list of 3-element tuples (oid, _==(x,y), f==label) -- just step through oid & f in ds and construct the dict comprehension self.what
        -self.what= {...., 4999:'obj0004999XX....XXX'  }  <-- last actual entry, note oid:f==label pairs in this dictionary
    -instance variable self.idx uses a dict comprehension to build {c==(x,y): oid}
        -remember -- ds is a list of 3-element tuples (oid, _c==(x,y), _==label) -- just step through oid & c in ds and construct the dict comprehension self.idx
        -self.idx= {....,  (93225264, 70034202): 4999   }<-- last actual entry, note c==(x,y):oid pairs in this dictionary
        -note that self.idx is a reverse index of oid:where to where:oid
    -actually, this class essentially takes in ds and builds up self.where, self.what, and self.idx dictionaries and _map and self.coord_map and has
       methods which either return (x,y) for an oid, or return oid for an (x,y)

    """

    def __init__(self, ds: List[Tuple[int, Tuple[int, int], str]]):  #ds = list of n objects in the gridxgrid array produced by gen_ds() above
        self.where: Dict[int, Tuple[int, int]] = {oid: c for oid, c, _ in ds}  # {...., 4999:(93225264, 70034202)  }
        self.what: Dict[int, str] = {oid: f for oid, _, f in ds} #{...., 4999:'obj0004999XX....XXX'  }
        self.idx: Dict[Tuple[int, int], int] = {c: oid for oid, c, _ in ds} #{....,  (93225264, 70034202): 4999   }

    def oid_to_coord(self, oid: int) -> Tuple[int, int]:
        """like in the Unified class, returns the *(x, y)* coordinate of object *oid*, except returning it from the "where" table
        --self.where dict supports oid input and get the (x,y) as output, i.e., what --> where
        -- this is actual last dictionary element in self.where
             {...., 4999:(93225264, 70034202)  }
             self.where[oid] e.g., self.where[0] --> (51706749, 56448162)
        """
        return self.where[oid]

    def coord_to_oid(self, coord: Tuple[int, int]) -> int:
        """returns the object id located at *coord* using the reverse index of self.where which is self.idx
        -note that self.idx is a reverse index of oid:where to where:oid
        -self.idx= {....,  (93225264, 70034202): 4999   }<-- last actual entry, note c==(x,y):oid pairs in this dictionary
        -actual returns of this method:
                e.g, coord_to_oid((29589712, 49937073)) --> 73
                e.g.,coord_to_oid((88662672, 77983095)) --> 2872
        """
        return self.idx[coord]

    def full_lookup(self, oid: int) -> Tuple[Tuple[int, int], str]:
        """return the combined (x, y) and feature string for *oid*
        -essentially it is the an integrated query call—it lets the benchmark fetch both the dorsal and ventral information
        about an object in one step, which mirrors in real life where a mammal would need the complete package (location and identity)
        -unified architecture can return both pieces from one table; the split architecture needs an explicit helper that
        joins its where and what tables to make the comparison fair; integrated query must trigger one public method in both architectures so that
        call overhead is identical, the difference measured reflects memory layout, not wrapper bookkeeping; unified can satisfy that method from
        one table; Split, by definition, cannot—so it needs a helper that explicitly joins its two stores inside the same call
        --> to keep wrapper bookkeeping fair for both models, there is one public method of call in both architectures (==models), the full_lookup method
        whether in Unified or in Split class -- however, in the Unified class' instance, that single call only requires one dictionary lookup inside
        the method while in the Split class' instance, it actually must trigger *two* dictionary look-ups (where and what) plus the integrity check reverse-index check
        --> both Unified and Split version of full_lookup have 1 call (and thus similar call overhead) but Unified has a unified lookup within the class,
        while Split version of pickup must do at least two times the work (acutally two dict look-ups plus the reverse-index check) within the class which reflects as best
        as possible the cost of keeping memories separate as occurs in the two-stream dorsal/ventral split hypothesis models
        -will call self.where  e.g., {...., 4999:(93225264, 70034202)  }
        -will call self.what   e.g., {...., 4999:'obj0004999XX....XXX'  }
        -will call self.idx    e.g., {....,  (93225264, 70034202): 4999   }
        -note that the lines oid_2 and the integrity check both can be deleted without affecting program functioning; they are there to keep the integrity of the joint dict's
        """
        c: Tuple[int, int] = self.where[oid]   #self.where  e.g., {...., 4999:(93225264, 70034202)  }  --> (93225264, 70034202)
        f: str = self.what[oid] #self.what  e.g., {...., 4999:'obj0004999XX....XXX'  } --> 'obj0004999XX....XXX'
        oid_2: int = self.idx[c]  #self.idx    e.g., {....,  (93225264, 70034202): 4999   }  --> 4999    nb. integrity check which can be deleted
        _ = self.what[oid_2]  #will raise KeyError if the "what" table lacks an entry; note we don't need its output so it is discarded nb. integrity check which can be deleted
        #print(c, f)  #e.g., (55687490, 20597897), 'obj0002395XXX...XXXX'
        return c, f


##Section 4 -- Benchmarking the Unified vs Split Model
##----------------------------------------------------
#
# Here's where the story gets interesting. We will have a Unified character (i.e., the Unified model) who will fight it out (well... in a peaceful,
#   constructive fashion :) with the Split character (i.e., the Split model).  Who will win?  What will the ending be like?
# Ok... sometimes in a complicated story you have to and look further in the book for a few moments. Below you will see a bunch
#   of benchmarking functions, but actually, when I am writing the code, I am creating the main() function at the same time --
#   main() is an overview of the storyline. So... let's just peek ahead and see what happens in main().
# First thing, scroll to the end of the code... we see "if __name__ == "__main__": main()" ... yup (slang for yes) main() will
#   be directing our story.
# Ok, thus we will next quickly scroll to "def main():"  The first code blocks are essentially getting command-line
#   arguments (or using defaults) and then assigning a bunch of parameters. But then we come to the function
#   "one(  )" -- aha (means "moment of sudden understanding") .... something's happening here.
# Ok, if we come back here and look below, right after "def run()" we see "def one()". This is where the interesting
#   action will start taking place.
# Ok, let's consider run() as some background to our story, and then we will jump into seeing all the action
#   that one() is doing.  This might be exciting :)
# And indeed, run() does do some vital operations -- it runs the specified number of queries in the specified probability mix of
#   integrated queries, what queries and where queries, and it times how long it takes for a WHAT lookup, a WHERE lookup and an INTEGRATED
#   lookup. It accumulates the time then returns the averaged per queries time in microseconds back to the calling method.
# Thus, at this point, we have a run() function that we can specify to whether it should use the Unified or Split class, we can specify the
#   probability mix (i.e, what % of queries should be integrated) and we can specify how many queries to run. And then it return to the calling
#   function the average latency in microseconds per query.
# Below the run() function you can see the one() function. one() pulls everything together into a single experimental trial
#    -it receives number of queries to run, the mix probabilities, the grid x grid size, and the Unified vs Split model to choose
#   -it will build the dataset ds and specify the Unified vs the Split class to use as well as number of queries and the % of integrated queries
#      to the run() function, and then the run() function reports the average latency time per query in microseconds
#   -one() will then return to its calling function a dictionary with the essential information of the experimental trial -- what
#      seed was used, what architecture (Split vs Unified) was used, and the latency time in microseconds.
# So at this point we have a function that can run experimental trials for different conditions which we can use to test the hypotheses
#   raised in the accompanying paper, i.e., under what conditions is the Split model better performing, if ever, than the
#   Unified model.
# As expected, main() will call one() -- main() can oversee all sorts of experiments by varying the parameters. However, how do we test the
#   hypothesis in the paper about parallelism?  Let's take a look at the following section, which is Section 5.


def run(mem: Union[Unified, Split], total_queries: int, mix: Dict[str, float], *, rng: random.Random) -> float:
    """micro-benchmark core function
    -this function sends a synthetic workload at both memory architectures (Unified and Split) and then averages the measurments of
       how long queries take on Unified and Split
    -it is called by one() and in turn by main() which repeat the above with different random trials, different query mixes and
       different dataset sizes
    -we use time.perf_counter_ns to measure, calculate and return mean latency in microseconds
    -the results measured here are what we are looking for in evaluating the Split versus the Unified architecture

    input parameters:
    - mem -- passed by one() to set up a Unified or a Split instance
          -- in one(), mem = Unified(ds) or else mem = Split(ds) -- this is
                what mem represents, essentially an instantiation of a class, in this case it
                represents an instance of class Unified of class Split depending how one instantiated it
    - total_queries -- number of random queries to execute in this run
                    -- passed by one() which in turn receives its own total queries from main() where it is args.q with a default at this writing of 10,000
    - mix -- probabilities that choose the query type, i.e., what % of queries are integrated queries
          -- forwarded by one() which in turn receives its own mix from main()
          -- it is a dictionary -  mix = {"integrated": p, "w2what": (1 - p) / 2, "what2w": (1 - p) / 2}
          -- the probability p values can be swept in main() e.g.,for p in (0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.35, 0.40, 0.50):
          -- all 3 p's in this dictionary must sum up to 1, i.e., p + (1-p)/2 + (1-p)/2 = 100%
          -- "w2what" where->what?  "what2w" what->where?  "integrated" full lookup
    - rng -- comes from one()
          -- in one() -- rng = random.Random(seed)
          -- "*,rng" was discussed above in gen_ds() --
                *, rng -- forces code calling this method to name the rng argument, e.g., gen_ds(1000, 5000, rng=my_rng) is ok
                note to the reader: largely taken from random library specs; do not give the *,rng structure much attention)
                rng must be seeded RNG (random number generator) object, e.g., my_rng = random.Random(42)
                   then it is ok to put rng.randint()

    returns:
    - acc/ total_queries /1000, i.e., running total of elapsed nanoseconds for all queries done divided by the number of queries
             divided by 1000 to convert nanoseconds to microseconds  --> mean latency time in microseconds for the mix of queries just run
    """
    #size = how many objects are in the dataset, and use this as the upper range value for oid's from 0 to size
    #then we create ids as list oid's (and will use again below) (more intuitive to use with mem.oid_to_coord(oid) like this)
    #then a quick list comprehension to create coords list of the where coordinates corresponding to those oid's*
    #remember -- regardless if Split or Unified, instance.oid_to_coord(oid) --> c
    if isinstance(mem, Split): #recall, mem -- passed by one() to set up a Unified or a Split instance
        size = len(mem.where)  #len(Split.where) = len({...., 4999:(93225264, 70034202)   })
        #print(size) #e.g., 5000 key-value pairs
    else:
        size = len(mem.id_map) #len(Unified.id_map) = len({..., oid:((x,y), 'objXXXX')})
    ids: List[int] = list(range(size)) #if size = 5000 then ids = [0, 1, ...., 4999]
    coords: List[Tuple[int, int]] = [mem.oid_to_coord(i) for i in ids] #e.g., [.....,  (93225264, 70034202)]

    #time.per_counter_ns() returns the current value of the high-resolution timer in nanoseconds
    nano_clock = time.perf_counter_ns  #no parentheses -- we are simply using nano_clock as an alias to reference this function, and avoid attribute lookup's as we loop below
    acc: int = 0  #we will use this as the accumulated running total of elapse nanoseconds for all queries in the run
    for _ in range(total_queries):  #recall, total_queries -- number of random queries to execute in this run
        r = rng.random() #random float from 0 to 0.999999999 - e.g., 0.765764358931739  since we will compare to probabilities p
        if r < mix["w2what"]:  #e.g., r 0.76576 < w2what 40% ?  False
            c = rng.choice(coords) #choose random coordinates from [coords]
            t0 = nano_clock() #starting time
            mem.coord_to_oid(c) #lookup oid from these coordinates  WHERE_TO_WHAT
            acc += nano_clock() - t0 #accumulated time for the operation
        elif r < mix["w2what"] + mix["what2w"]: #e.g., r 0.76576 < w2what 40% + what2w 40% = 80%?  True
            o = rng.choice(ids) #choose a random oid from [ids], e.g., 1812
            t0 = nano_clock() #e.g., 44887206183000 nsecs starting time
            mem.oid_to_coord(o) #look up coordinates for this oid  WHAT_TO_WHERE
            acc += nano_clock() - t0  #e.g., 600 nsecs
        else:
            o = rng.choice(ids) #choose random oid from [ids]
            t0 = nano_clock() #starting time
            mem.full_lookup(o) #lookup of oid, f (which takes different numbers of dict calls as noted above in Unified versus Split)
            acc += nano_clock() - t0 #accumulated time for this branch's operation

    #at this point we have looped through this random assignment of queries per the probabilities in [mix], and kept accumulating the
    #  time in nanosceconds it has taken for the operations
    #as noted above, we divide this accumulated total acc by the number of queries we ran and then by 1000 to convert to
    #  microseconds  --> mean latency time in microseconds for the mix of queries just run is returned
    return acc / total_queries / 1000


def one(seed: int, n: int, total_queries: int, mix: Dict[str, float], grid: int, model_kind: str) -> Dict[str, Any]:
    """one() pulls everything together into a single experimental trial
    -it receives number of queries to run, the mix probabilities, the grid x grid size, and the Unified vs Split model to choose
    -it will build the dataset ds
    -it will choose a Unified or a Split model memory architecture, i.e., from its parameters
    -it will then pass these parameters to run() for a timing benchmark
    -it then returns the result of this experimental trial back to the calling funcion

    -main() calls one() to run different experimental trials with different random trials, mixes of percentage probability of integrated questions,
        different numbers of queries, different numbers of objects chosen and different grid x grid sizes used
    -in the multiprocessing functions, _worker() calls one() to allow each subprocess time its own copy of the benchmark, i.e., where we
        examine the effect of parallelism on the differences between the models

    input parameters:
    -seed  - a random seed for random.Random which gives rng, and then rng is required for gen_ds() as discussed earlier
    -n - number of objects in the trial
    -total_queries - number of queries in the trial
    -mix - a dictionary as discussed above showing the probability percentages of integrated queries, what queries and where queries in the trial
    -grid - the (x,y) locations of the objects will be randomly placed within a grid x grid array in the trial
    -model_kind - specifies whether the trial tests the Unified or Split memory model

    returns:
    -a dictionary showing {"seed":seed, "arch":model_kind, "mean_us":runtime in usecs} where
       runtime in usecs = run(mem, total_queries, mix, rng=rng)
       -this dictionary can be used later with Pandas when we decide how to display the data which supports or contradicts hypotheses of the associated paper
    """
    rng = random.Random(seed)  #random seed required by gen_ds() as discussed above
    ds = gen_ds(n, grid, rng=rng) #generate the dataset for this experimental trial with n objects and gridxgrid coordinate space
    mem: Union[Unified, Split] #required for mypy later to let it know either of these types are ok
    if model_kind == "Unified":
        mem = Unified(ds)  #depending on model_kind parameter, mem is an instance of the Unified class
    elif model_kind == "Split":
        mem = Split(ds) #depending on model_kind parameter, mem is an instance of the Split class
    else:
        raise ValueError(f"devp't error: Non-Unified/Split architecture {model_kind} specified as 'model_kind' parameter in method one()")
    #as noted above one() calls run() with parameters for mem (Unified or Split model), total_queries to run, mix of integrated and other queries %,
    #one() then returns a dict with {seed, model_kind, latency in microseconds of the experimental trial}, i.e., the results of the experimental trial
    return {"seed": seed, "arch": model_kind, "mean_us": run(mem, total_queries, mix, rng=rng)}


##Section 5 -- A Mix of Statistics, Formatting and Multiprocessing
##----------------------------------------------------------------
#
# Often in the middle of a novel there may be a stretch of a little bit less exciting material, that really serves to take
#  the storyline towards its conclusion. In a way, that's what's happening here.
# In this section, there are number of functions which help with statistics, formatting and multiprocessing. You don't need to
#  dwell on many of the fine details here -- I am largely following recommended specifications of the libraries I am using for
#  these features, albeit, as applied specifically to the purposes of this program.
# The functions format_dataframe(), wide_dataframe(), and stringify_dataframe() essentially help to format the data for the Pandas library to display
#  the results of our experimental trials neatly in tables on the monitor, and to do some simple arithmetic and statistical calculations. These functions
#  are all called by main() (which I am writing at the same time as I write these functions -- the reason the code is here and not in
#  main() is to abstract away details from main() so that you the reader can make better sense of main() )
# The functions parallel_benchmark() and _worker() set up the parallel benchmarking. In the accompanying paper there is the hypothesis that
#  parallelism will affect the performance of Split versus Unified models, so we need to test this out somehow in our simulation. As noted above
#  in the imports section and below, due to the Python GIL lock we really need to set up separate processes using their own version of Python, and
#  we use the multiprocessing "mp" module to do this for us. (Actually, it does a great job of abstracting away all the details of multiprocessing -- little
#  expertise is required in this area -- we simply use the mp module and let it manage all the details of spawning and running multiple processes.)
# The function parallel_benchmark() is called by main() to test parallel operation of the Unified versus Split models. This function uses mp to set up
#  separate worker processes which will run query tasks on the Unified and Split models. As before we use the one() function to run these
#  query tasks. Although one() could have been called directly from parallel_benchmark() I was required to set up _worker(). This is interesting --
# in order to use it with pool.map() -- it needs a function name that takes a single argument. Thus, _worker() with its tuple input parameter
# was created.
# The function start_multiprocessing() simply abstracts away from main() some of the start up activities of using multiprocessing, and offers future
#  development the opportunity to better use multiprocessing.
# Ok... so at this point in the story, we now have set up our main characters of the story, i.e., the instances of the Unified and the Split
#  classes. We have set up query tasks for them to perform and to be benchmarked on. We have set up the one() function to orchestrate experimental
#  trials of different parameters. And now, very importantly, we have set up a multiprocessing mechanism to access the one() function() to test out
#  the hypotheses in the accompanying paper about the effect of parallelism on Split versus the Unified model.
# This is very exciting!! In the next section, let's pull everything together and let our Split model compete with the Unified model (albeit,
#   in a peaceful, constructive fashion:) and let's see what happens.


def format_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Turns the dicts we created in each trial in one() into a table that groups by architecture and shows basic statistics
    -we will calculate mean and standard deviations of the numerical information
    -main() calls format_dataframe()

    input parameters:
    -rows - output of one() {seed, arch, latency} for a number of experiments
      e.g., [{'seed': 0, 'arch': 'Unified', 'mean_us': 0.11225}, ....., {'seed': 49, 'arch': 'Split', 'mean_us': 0.12707}]
    returns:
    -Pandas dataframe, i.e., pd.DataFrame with settings to print the table as shown below
    """
    df = pd.DataFrame(rows)
    pp = (df.groupby("arch")["mean_us"].agg(["mean", "std"])
        .rename(columns={"mean": "μ (µs)", "std": "σ (µs)"}))
    pp["Δ vs Unified (%)"] = ((pp.loc["Unified", "μ (µs)"] - pp["μ (µs)"]) / pp.loc["Unified", "μ (µs)"] * 100).round(1)
    return pp


def wide_dataframe(df: pd.DataFrame, *, higher_is_better: bool = False) -> pd.DataFrame:
    """Converts a DataFrame where index holds the sweep dimension, columns are "Split" and "Unified", into a wider format
        with an extra Δ column
    -this could have been included with format_dataframe() but set up for flexibility with different potential tables and
       future usage, particularly if want to various statistical operations; consider combining the two functions in the future
    -main() calls wide_dataframe()

    input parameters:
    -df -- a Pandas DataFrame whose columns are exactly Split, Unified
    -higher_is_better -- a flag denoting whether a higher or lower number should be considered better in the table
                      -- note that it is defined as a keyword-only parameter (i.e., it is after the asterisk in the definition)
    returns:
    -Pandas dataframe, i.e., pd.DataFrame identical to df but with a new column "delta vs Unified %"
    """
    out = df.copy()
    if higher_is_better:
        out["Δ vs Unified (%)"] = ((out["Split"] - out["Unified"]) / out["Unified"] * 100).round(1)
    else:
        out["Δ vs Unified (%)"] = ((out["Unified"] - out["Split"]) / out["Unified"] * 100).round(1)
    return out


def stringify_dataframe(title: str, df: pd.DataFrame) -> None:
    """This is essentially a lightweight wrapper to print a header line and a DataFrame
        using the same float format each time
    -main() calls strinigy_dataframe()

    input parameters:
    -title -- section heading
    -df  -- Pandas dataframe to be stringified
    returns:
    -none (function prints to stdout)
    ."""
    print(f"\n=== {title} ===")
    print(df.to_string(float_format="%.3f"))


def print_welcome():
    """Welcome message to user when the program starts.

    -called by main()
    -no input parameters
    -no return value
    """
    print('''
    \nSimulation to accompany paper: The Two-Stream Hypothesis as a Foundation for Human-like Memory and Action
    Howard Schneider, May 2025  hschneidermd@alum.mit.edu

    In the two-stream hypothesis, the mammalian brain’s visual (and possibly other sensory) information diverges
    into a ventral “what” pathway for object identity and a dorsal “where” pathway for spatial planning. Although
    this split is well documented, its computational value and its role in the emergence of human-like
    intelligence remain debated. We combine a neurobiological review with targeted simulations to evaluate
    when a split model (i.e., “what” and “where” facts are stored separately) outperforms a unified model (i.e.,
    facts stored together). This program is intended to create the targested simulations.

    The results of the simulation are below. Note that while absolute timings vary by hardware, the qualitative
    trends should remain robust. Even on the same hardware there are stochastic considerations which will effect
    results, as well as the effect of a myriad of background processes that the hardware's operating system
    is managing. As noted below, for example, just because 16 worker processes are specified, this does not
    mean that 16 cores (if possible on a given computer, for example) will be dedicated to these worker processes.

    Note that you can specify certain parameters on the command line if you wish. To see what you can specify
    enter on your terminal:        >python name_of_program.py -h\n
    ''')

    #no return value


def start_multiprocessing():
    """The function start_multiprocessing() simply abstracts away from main() some of the start up activities of using multiprocessing,
       and offers future development the opportunity to better use multiprocessing.

    called by main() before multiprocessing is started

    input parameters:
    - none

    returns:
    - none
    """
    mp.freeze_support()
    #this mp code is only required when the Python script is converted to a Windows executable, e.g., with PyInstaller -- otherwise it will be ignored
    print(f"\nThis CPU has the following number of logical cores theoretically available at this time: {os.cpu_count()}")
    print("(The OS scheduler will actually decide which cores to assign Python processes to and there may not always")
    print("  be a dedicated core assigned to each process. As well, if there are more processes than cores, then the")
    print("  OS will time-slice which may introduce other delays.)")
    #no values returned; end of function


def _worker(gg: Tuple[str, int, int, int, Dict[str, float], int]) -> float:
    """This is essentially a small wrapper so that mutliprocessing.Pool.map() can use a single function
        that will call one() and returns just the mean latency time
    -the name comes from organzing worker processes in examining class performance under different parallelism
    -quick note on terminology used here:
       -- the underscore _worker() simply means this function should be used only by parallel_benchmark(), i.e., we are using standard python
            notation to flag it as conventionally private, but it has no effect on its operation
       -- any spawned process by multiprocessing module can be called a "process" or can be called a "worker process" if the process does the work,
            hence my frequent use of the term "worker process" or "worker" for short
       -- the term "worker subprocess" means the exact same thing really, but is somewhat redundant
       -- the term "subprocess" is sometimes used interchangeably with process as I have done here; however, there is a subprocess module for shell
            commands so I really should not be using it and I will clean up the docstrings in the future, but where I used it, just consider it the same as process

    parallel_benchmark() calls the function within the subprocess

    input parameters:
    gg is a tuple which is packed with the following parameters based on parameter order:
    -arch -- "Unified" or "Split"
    -seed --  random seed for this worker subprocess
    -n -- number of ojbects in the dataset
    -total_queries -- number of queries to ask
    -mix -- the % of integrated queries
    -grid -- the grid x grid coordinate space to set up

    returns:
    -mean_us -- mean latency time in microseconds for that subprocess
    -does not return any of the other values from the results dictionary

    """
    arch, seed, n, total_queries, mix, grid = gg
    return one(seed, n, total_queries, mix, grid, arch)["mean_us"]  # type: ignore[index]


def parallel_benchmark(worker_count: int, n: int, total_queries: int, mix: Dict[str, float], grid: int) -> Tuple[float, float]:
    """This function runs the benchmark in parallel on multiple worker processes, and then computes the
        total kilo-queries per second that occurred

    called by main() when parallelism is tested

    input parameters:
    -worker_count -- worker process count (1,4,8,16)
    -n - objects to include in the grid x grid
    -total_queries -- total queries each work process must do
       --therefore, the total number of queries will be worker_count * total_queries
    -mix -- the dictionary specifying the % of integrated and other what and where questions
    -grid -- the size of the grid x grid coordinate space

    returns:
    -(kps for Split, kps for Unified)

    -essentially a few lines of code is able to test the Split versus the Unified model with regard to
       parallel operation, where we compare performance running on different number of separate Python interpreters
       in parallel
    -the function makes use of Python's standard library multiprocesing module, and follows the examples provided by
      the official Python documentation of the module
    -above multiprocessing is imported as mp
    -as noted above in the imports section, we will run separate processes each with its own Python interpreter and thus bypass the issue of
       GIL lock if we had instead run multi-threads implementation
    -multiprocessing is a package that allows us to spawn processes
    -consider a simple example:
       def square(x):
          return x*x
       with mp.Pool(4) as aa:  # <--Pool(4) creates 4 workers processes
          results = aa.map(square, [1,2,3,4,5])
       print(results)
    -each of the 4 worker processes will operate on part of the input list of [1,2,3,4,5] and return its results to Pool.map() which then
       returns a collected [results] -- we don't have to worry about the internal operations, the multiprocessing library will do it for us;
       effectively this is the same result if no multiprocessing was done, but it can be done faster in theory since 4 worker processes can do it
       instead of one single process; on some PC CPU's there may be multiple cores so each core can do part of the work in parallel; mp.Poolmap()
       collects all the results and reassembles them in the same order as the input list -- it abstracts away all the details of the multiprocessing
    -as noted at the start of this program, I was not going to use lambdas, functional programming, etc in order to keep the code at the level of
       intermediate Python, so that it would be readable to most readers; although the map() function borrows some ideas from functional programming,
       it really is not functional in reality; the reason I am using it here is because it is easier to use the multiprocessing library as, and is
       given as an example in the first paragraph of the official Python documentation of the multiprocessing module
    -given this small tutorial on "mp" (i.e., multiprocessing module) you can see in the coode below, we do essentially the sam thing -- we set up
       multiple worker subprocesses via "with mp.Pool(worker_count) as pool:", and then we use the same "pool.map()" function as in the simple example
       above to let mp behind the scenes use different worker processes to split up the work subtasks can call the _worker function (essentially wrapper)
       which calls one() to get the latency:  one(seed, n, total_queries, mix, grid, arch)["mean_us"] which is returned to pool.map without us having
       to worry about the details; actually we are just interested in the time taken which is what we measure and get our kilo-queries per second benchmark

    """
    tot = worker_count * total_queries  #total queries by all the worker processes running
    kps: Dict[str, float] = {} #store kilo-queries per second results for each model in this dict
    for arch in ("Split", "Unified"): #make sure it is either Split or Unified architecture
        subtasks = [(arch, i, n, total_queries, mix, grid) for i in range(worker_count)]
        #build an argument list substasks which is made up of tuples as shown
        #e.g.,[('Split', 0, 50000, {'integrated': 0.25, 'w2what': 0.375, 'what2w': 0.375}, 1000000)]
        t0 = time.perf_counter() #start time with time from normal second clock (not the nanosecond clock as above)
        with mp.Pool(worker_count) as pool: #spawn a process pool of size worker_count
            #mp.Pool is multiprocessing.Pool that spawns child processes
            #pool.map distributes items round-robin to the worker_count processes in the pool
            #each subprocess has its own Python interpreter so queries can run in parallel
            pool.map(_worker, subtasks)  #each subprocess calls _worker above which calls one() for an experimental trial
            #note: as mentioned at the start of the program, there is no real functional programming or advanced techniques in this code
            #pool.map is the most basic way of using the mp module per the Python official documentation; just think of it as splitting
            #  up the input list to a number of worker processes running on their own, and then pool.map automatically collects the partial
            #  work and assembles all the results in for us -- mp abstracts away all the details of the multiprocessing
            #very important -- note that we essentially discard the results returned by _worker() -- we are just interested in how long
            #  this all takes so that we can compute a kilo-queries per second benchmark
        kps[arch] = tot / (time.perf_counter() - t0) / 1000 #build up kps dictionary for each of the arch
    return kps["Split"], kps["Unified"]

##END FUNCTIONS, CLASSES


##START INTRO-MAIN

##Section 6 -- The Big Competition: the Split Model versus the Unified Model
##--------------------------------------------------------------------------
#
# Ok... as noted at the end of the last section, at this point in the story, we now have set up our main characters of the story,
#   i.e., the instances of the Unified and the Split classes.
# We have also set up query tasks for them to perform and to be benchmarked on. We have set up the one() function to orchestrate experimental
#  trials of different parameters. And we have also set up a multiprocessing mechanism to access the one() function() to test out the
#  effect of parallelism on Split versus the Unified model.
# Now let's pull everything together and let the competition begin... the Split model versus the Unified model!!
# Hmmm....
# Ok... actually... we need to set up what parameters we want to test before the competition can begin. We will be guided by the hypotheses in
#  the accompanying paper and set these up. For the moment let's do this in the main() function since these conditions are really top-level
#  questions we are asking, as per the paper. However, if main() gets cluttered up, we can later move some of this set up code to their own functions.
# The first part of main() is the welcome message and the default parameters.
# In the next part of main() we generate what the paper describes as the canonical benchmark, i.e., how many microseconds on average to answer a query
#   for the default conditions with 20% of the queries being integrated queries (i.e., both what and where info requested), and we do this for the
#   Split model and the Unified model. Faster, i.e., lower latency times, is considered better.
# Then we compare the latency times (lower is better) to answer queries by the Split model versus the Unified Model
#   for different parameter values for the % of integrated (what plus where info) queries of all the queries.
# Then we compare the latency times (lower is better) to answer queries by the Split model versus the Unified Model
#   for different parameter values of the number of objects in the simulation.
# Last but not least, we will let the Split Model compete with the Unified Model as we vary the number of worker processes
#   being used, and as such, test how parallelism affects the operating regimes of the models.
# The section "Parallel processes parameter sweep" should be straightforward to follow. We start up multiprocessing and set the mix of % integrated queries.
#   We then have a simple for loop where we try out 1, 4, 8 and 16 spawned worker processes running in parallel. In each for loop, we call the function
#   parallel_benchmark() and it will return the number of kiloqueries per second the Split model was able to accomplish and the number the Unified model
#   was able to accomplish. As each for loop is run, we save the results in the dictionary "parallel_results" and then we use Pandas to dispaly a table of
#   these results.
# Ok... so we've measured the results of the Split Model against the Unified Model when we swept the percentage of integrated questions, the number of
#   objects to handle, and the number of parallel processes used. What happened?? Who won??

def main() -> None:
    """
    -The first part of main() is the welcome message and the default parameters.
    -In the next part of main() we generate what the paper describes as the canonical benchmark, i.e., how many microseconds on average to answer a query
     for the default conditions with 20% of the queries being integrated queries (i.e., both what and where info requested), and we do this for the
     Split model and the Unified model. Faster, i.e., lower latency times, is considered better.
    -Note that we pass the selected parameters of this 'canonical benchamark' to the funciton one() which runs the experimental trials
      per these parameters and returns dictionary showing {"seed":seed, "arch":model_kind, "mean_us":runtime in usecs}
    -Note that we run one() with the given set of parameters over and over again -- args.trials (default 50) times-- for better statistical results
    -Then we compare the latency times (lower is better) to answer queries by the Split model versus the Unified Model
     for different parameter values for the % of integrated (what plus where info) queries of all the queries.
    -Again we do args.trials number of runs for each set of parameters, and pass the parameters to one()
    -Then we compare the latency times (lower is better) to answer queries by the Split model versus the Unified Model
     for different parameter values of the number of objects in the simulation.
    -Again we do args.trials number of runs for each set of parameters, and pass the parameters to one()
     Last but not least, we will let the Split Model compete with the Unified Model as we vary the number of worker processes
      being used, and as such, test how parallelism affects the operating regimes of the models.
    -The various parameters are sent to one() which returns latency time, but we ignore the actual times, but instead see how many
      queries can be processed in a second for the Split model versus the Unified model. The parameters including the number of worker processes
      to run in parallel are passed to the function parallel_benchmark() which as we noted above, uses the multiprocessing 'mp' module
      pool.map to distribute items to various worker processes who work on queries in parallel, and returns kiloqueries answered per second
      for the Split model and for the Unified model.
    -Pandas is used to do some statistical computation, format the results and display the results on the terminal.

    input parameters:
      -none (although can use argparse for parameter values)
    returns:
      -none

    """
    #welcome message and set up default parameters
    print_welcome()
    pa = argparse.ArgumentParser(description="argparse_for_two_stream_code")
    pa.add_argument("--n", type=int, default=5_000) #number of objects in coordinate space
    pa.add_argument("--q", type=int, default=10_000)#number of queries
    pa.add_argument("--trials", type=int, default=50) #number of times to call one() for non-parallel parameter sweeps for better statistical purposes
    pa.add_argument("--grid", type=int, default=100_000_000) #the grid x grid size of the coordinate space
    args = pa.parse_args()  #read command-line or fall back to defaults
    mix_qty: Dict[str, float] = {"w2what": 0.47, "what2w": 0.47, "integrated": 0.06}  #object sweep benchmark
    mix_canonical: Dict[str, float] = {"w2what": 0.4, "what2w": 0.4, "integrated": 0.2}  #canonical benchmark
    n_mix: int = 2_500 #mix sweep benchmark
    grid_mix: int = 300_000_000 #mix sweep benchmark
    scale_n: int = 20_000 #parallel benchmark
    scale_grid: int = 1_000_000 #parallel benchmark
    scale_q: int = 50_000  #parallel benchmark

    # performance of Split vs Unified models for default conditions with 20% integrated queries
    # "canonical benchmark" in the accompanying paper
    rows: List[Dict[str, Any]] = []
    for trial in range(args.trials): #more trials, then more repeat of the benchmark
        for model in ("Unified", "Split"):
            rows.append(one(trial, args.n, args.q, mix_canonical, args.grid, model)) #we append the results of each run to rows[]
    #print(rows)  #e.g., rows = [{'seed': 0, 'arch': 'Unified', ...., 'mean_us': 0.12356}, {'seed': 49, 'arch': 'Split', 'mean_us': 0.13915}]
    stringify_dataframe("Split vs Unified Model Latency Time for Canonical 40/40/20 Query Mix", format_dataframe(rows)) #use Pandas to print out a table of the results

    # performance of Split vs Unified models for a sweep of the mix (i.e., % integrated queries) parameter
    mix_rows: List[Dict[str, Any]] = []
    for prob_integrated in (0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.35, 0.40, 0.50): #% integrated queries
        mix = {"integrated": prob_integrated, "w2what": (1 - prob_integrated) / 2, "what2w": (1 - prob_integrated) / 2} #this is mix parameter we will vary in the experimental trials via one()
        for trial in range(args.trials): #more trials, then more repeat of the benchmark
            for model in ("Unified", "Split"): #experimental trial on Unified and then on Split model
                rec = one(trial, n_mix, args.q, mix, grid_mix, model) #one() runs the experimental trial for a given set of parameters
                #print(rec) # e.g., rec = {'seed': 0, 'arch': 'Unified', 'mean_us': 0.10205}
                rec["p"] = prob_integrated
                #print (rec) e.g., rec = {'seed': 0, 'arch': 'Unified', 'mean_us': 0.10205, 'p': 0}
                mix_rows.append(rec)
    # print(mix_rows) #e.g., mix_rows = [{'seed': 0, 'arch': 'Unified', 'mean_us': 0.10205, 'p': 0}, ...., {'seed': 49, 'arch': 'Split', 'mean_us': 0.13975, 'p': 0.5}]
    df_mix = pd.DataFrame(mix_rows).groupby(["p", "arch"])["mean_us"].mean().unstack() #use Pandas to print out a table of the results
    df_mix.index = (df_mix.index * 100).astype(int).astype(str) + "% integrated"
    stringify_dataframe("Split vs Unified Latency Time (us) for Mix Parameter Sweep", wide_dataframe(df_mix))

    # performance of Split vs Unified models for a sweep of the number of objects parameter
    sizes = [100, 1_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000] #number of objects we will vary in the experimental trials via one()
    qty: List[Dict[str, Any]] = []
    for number_objects in sizes:
        for trial in range(args.trials):  #more trials, then more repeat of the benchmark
            for model in ("Unified", "Split"): #experimental trial on Unified and then on Split model
                rec = one(trial, number_objects, args.q, mix_qty, args.grid, model) #one() runs the experimental trial for a given set of parameters
                rec["n"] = number_objects #results of this experimental trial
                qty.append(rec) #append results to qty for all the results
                #print(rec) #e.g., rec = {'seed': 0, 'arch': 'Unified', 'mean_us': 0.09348, 'n': 100}
                #print(qty) #e.g., qty = [{'seed': 0, 'arch': 'Unified', 'mean_us': 0.09348, 'n': 100}, {'seed': 0, 'arch': 'Split', 'mean_us': 0.24839, 'n': 100}, ....]
    df_objects = pd.DataFrame(qty).groupby(["n", "arch"])["mean_us"].mean().unstack() #use Pandas to print out a table of the results
    stringify_dataframe("Split vs Unified Latency Time (us) for Object Quantity Sweep", wide_dataframe(df_objects))

    # performance of Split vs Unified models for a sweep of the number of parallel worker processes used
    start_multiprocessing()  #start up activities required to use multiprocessing
    mix25: Dict[str, float] = {"integrated": 0.25, "w2what": 0.375, "what2w": 0.375} #speciyf proportion of integrated queries
    parallel_results: Dict[str, List[float]] = {}
    #run parallel_benchmark() for varying number of worker processes
    for number_workers in (1, 4, 8, 16): #number of worker processes to spawn and run the benchmark on
        #parallel_benchmark() will return kps["Split"], kps["Unified"] -- we will store  these results as dict parallel
        s_kqps, u_kqps = parallel_benchmark(number_workers, scale_n, scale_q, mix25, scale_grid) #split kiloqueries/sec, unified kiloqueries/sec
        parallel_results[f"{number_workers} worker(s)"] = [s_kqps, u_kqps]
        #as the loop runs, the parallel_results{} dictionary keys become "1 worker(s)", "4 workers(s)", etc
        #the value of any given key is given by the list [s_kqps, u_kqps]  (hint: look at the typing information to quickly see what I am trying to do)
        #actual example from a run, e.g., parallel_results = {'1 worker(s)': [114.782, 108.943], ...., '16 worker(s)': [538.488, 496.229]}
    #use Pandas to print a table of the results on the terminal
    dfP = pd.DataFrame(parallel_results).T.rename(columns={0: "Split kQ/s", 1: "Unified kQ/s"})
    stringify_dataframe(
        "Split vs Unified kiloqueries/sec with Sweep of Worker Processes ",
        wide_dataframe(dfP.rename(columns={"Split kQ/s": "Split", "Unified kQ/s": "Unified"}), higher_is_better=True))

##Section 7 -- The Big Competition: Who Won?
##------------------------------------------
#
# Ok... as we noted in the last section, we've measured the results of the Split Model against the Unified Model when we swept
#  the percentage of integrated questions, the number of objects to handle, and the number of parallel processes used.
#  What happened?? Who won??
# Actually, these results go into the accompanying paper. Here is the abstract to give you some of the details.
# TL;DR -- Simulations (n=5,000 objects) show that a split model answers integrated queries (object identity plus location)  ≈17% slower
#  than a unified model. While absolute timings vary by hardware, the qualitative trends remain robust. Parameter sweeps reveal
#  a crossover: the split model becomes more efficient when integrated queries fall below ≈15% of the total
#  (> 85% ask only “what” or “where”), when object memory exceeds cache efficiency, or when there is more parallelism.
# Full abstract:
# In the two-stream hypothesis, the mammalian brain’s visual (and possibly other sensory) information diverges into a ventral
#“what” pathway for object identity and a dorsal “where” pathway for spatial planning. Although this anatomical split is well documented,
#its computational value and its role in the emergence of human-like intelligence remain debated. We combine a neurobiological review
#with targeted simulations to evaluate when a split model (i.e., “what” and “where” facts are stored separately) outperforms a unified
#model (i.e., facts stored together).  Simulations (n=5,000 objects) show that a split model answers integrated queries (object identity
#plus location)  ≈17% slower than a unified model. While absolute timings vary by hardware, the qualitative trends remain robust.
#Parameter sweeps reveal a crossover: the split model becomes more efficient when integrated queries fall below ≈15% of the total
#(> 85% ask only “what” or “where”), when object memory exceeds cache efficiency, or when there is more parallelism. We argue that
#this operating regime matches real-world conditions in which rapid visuomotor loops and abstract conceptual learning proceed
#simultaneously, and that the forced “binding step” between streams may foster compositional thought. The results suggest that
#representational segregation is not an evolutionary relic but a computationally advantageous principle worth implementing in
#brain-inspired cognitive architectures (BICAs).

if __name__ == "__main__":
    main()
##END INTRO-MAIN
