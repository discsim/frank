"""Provides a list of quotes
"""
from __future__ import division, absolute_import, print_function

import random

__all__  = [ "get_quote", "print_quote" ]


# TODO: These are stolen from KROME and should be replaced
_quotes =  [['', ''],
            ["If you lie to the computer, it will get you.","Perry Farrar"],
	    ["Premature optimization is the root of all evil.",
             "Donald Knuth"],
	    ["Computers are good at following instructions, but not at "
             "reading your mind.","Donald Knuth"],
	    ["Computer Science is embarrassed by the computer.",
             "Alan Perlis"],
	    ["Prolonged contact with the computer turns mathematicians into "
             "clerks and vice versa.","Alan Perlis"],
	    ["There are two ways to write error-free programs; only the "
             "third one works.","Alan Perlis"],
	    ["Software and cathedrals are much the same - first we build "
             "them, then we pray.","Sam Redwine"],
	    ["Estimate always goes wrong.","Sumit Agrawal"],
	    ["Weinberg's Second Law: If builders built buildings the way "
             "programmers wrote programs, then the first woodpecker that "
             "came along would destroy civilization.","Gerald Weinberg"],
	    ["Any sufficiently advanced magic is indistinguishable from a "
             "rigged demonstration.",""],
	    ["Any given program, when running, is obsolete.",""],
	    ["Programming would be so much easier without all the users.",""],
	    ["Your Zip file is open.",""],
	    ["Testing can only prove the presence of bugs, not their "
             "absence.","Edsger W. Dijkstra"],
	    ["If debugging is the process of removing bugs, then programming"
             " must be the process of putting them in.","Edsger W. Dijkstra"],
	    ["God is Real, unless declared Integer.","J. Allan Toogood"],
	    ["Curiously enough, the only thing that went through the mind of" 
             "the bowl of petunias as it fell was Oh no, not again.",
             "The Hitchhiker's Guide to the Galaxy"],
	    ["Computer science differs from physics in that it is not "
             "actually a science.","Richard Feynman"],
	    ["The purpose of computing is insight, not numbers.",
             "Richard Hamming"],
	    ["Computer science is neither mathematics nor electrical "
             "engineering.","Alan Perlis"],
	    ["I can't be as confident about computer science as I can about "
             "biology. Biology easily has 500 years of exciting problems to "
             "work on. It's at that level.","Donald Knuth"],
	    ["The only legitimate use of a computer is to play games.",
             "Eugene Jarvis"],
	    ["UNIX is user-friendly, it just chooses its friends.",
             "Andreas Bogk"],
	    ["Quantum mechanic Seth Lloyd says the universe is one giant, "
             "hackable computer. Let's hope it's not running Windows.",
             "Kevin Kelly"],
	    ["Computers are useless. They can only give you answers.",
             "Pablo Picasso"],
	    ["Computers in the future may weigh no more than 1.5 tons.",
             "Popular Mechanics (1949)"],
	    ["Don't trust a computer you can't throw out a window.",
             "Steve Wozniak"],
	    ["Computers are like bikinis. They save people a lot of "
             "guesswork.","Sam Ewing"],
	    ["If the automobile had followed the same development cycle as "
             "the computer, a Rolls-Royce would today cost $100, get a "
             "million miles per gallon, and explode once a year, killing "
             "everyone inside.","Robert X. Cringely"],
	    ["Computers are getting smarter all the time. Scientists tell us"
             " that soon they will be able to talk to us.  (And by 'they'"
	     " I mean 'computers'.  I doubt scientists will ever be able to "
             "talk to us.)","Dave Barry"],
	    ["Most software today is very much like an Egyptian pyramid with"
             " millions of bricks piled on top of each other, with no "
             "structural integrity, but just done by brute force and "
             "thousands of slaves.","Alan Kay"],
	    ["No matter how slick the demo is in rehearsal, when you do it "
             "in front of a live audience, the probability of a flawless "
	     "presentation is inversely proportional to the number of people"
             " watching, raised to the power of the amount of money "
             "involved.", "Mark Gibbs"],
	    ["Controlling complexity is the essence of computer "
             "programming.","Brian Kernigan"],
	    ["Software suppliers are trying to make their software packages "
             "more 'user-friendly'...  Their best approach so far has been "
             "to take all the old brochures and stamp the words "
             "'user-friendly' on the cover.","Bill Gates"],
	    ["Programmers are in a race with the Universe to create bigger "
             "and better idiot-proof programs, while the Universe is trying "
             "to create bigger and better idiots. So far the Universe is "
             "winning.","Rich Cook"],
	    ["To iterate is human, to recurse divine.","L. Peter Deutsch"],
	    ["Should array indices start at 0 or 1?  My compromise of 0.5 "
             "was rejected without, I thought, proper consideration.",
             "Stan Kelly-Bootle"],
	    ["Any code of your own that you haven't looked at for six or "
             "more months might as well have been written by someone else.",
             "Eagleson's Law"],
	    ["All science is either physics or stamp collecting.",
             "Ernest Rutherford"],
	    ["Done is better than perfect.", ""],
	    ["Computers are like Old Testament gods; lots of rules and no"
             " mercy","Joseph Campbell"],
	    ["A computer lets you make more mistakes faster than any other "
             "invention with the possible exceptions of handguns and "
             "Tequila.", "Mitch Ratcliffe"],
	    ["Computer Science is no more about computers than astronomy is "
             "about telescopes.","Edsger W. Dijkstra"],
	    ["To err is human, but to really foul things up you need a "
             "computer.","Paul Ehrlich"],
	    ["Debugging is twice as hard as writing the code in the first "
             "place. Therefore, if you write the code as cleverly as "
             "possible, you are, by definition, not smart enough to debug"
             " it.","Brian W. Kernighan"],
	    ["Always code as if the guy who ends up maintaining your code "
             "will be a violent psychopath who knows where you live.",
             "Martin Golding"],
	    ["One of my most productive days was throwing away 1000 lines of"
             " code.","Ken Thompson "],
	    ["And God said, \"Let there be light\" and segmentation fault "
             "(core dumped)",""],
	    ["Today, most software exists, not to solve a problem, but to "
             "interface with other software","Ian Angell"],
	    ["Measure twice, cut once",""],
	    ["Weeks of programming can save you hours of planning",""],
	    ["All models are wrong; some models are useful","George Box"],
	    ["The generation of random numbers is too important to be left "
             "to chance","Robert Coveyou"],
	    ["Problems worthy / of attack / prove their worth / by hitting "
             "back","Piet Hein"],
	    ["Good, Fast, Cheap: Pick any two","Memorandum RFC 1925"],
	    ["One size never fits all","Memorandum RFC 1925"],
	    ["No matter how hard you push and no matter what the priority, "
             "you can't increase the speed of light","Memorandum RFC 1925"],
	    ["Chemistry has been termed by the physicist as the messy part "
             "of physics", "Frederick Soddy "],
	    ["Don't worry if it doesn't work right. If everything did, you'd "
             "be out of a job.", "Mosher's Law"],
	    ["Beware of bugs in the above code; I have only proved it "
             "correct, not tried it.", "Donald Knuth"],
	    ["Given enough eyeballs, all bugs are shallow.",
             "Eric S. Raymond"],
	    ["We make an extreme, but wholly defensible, statement: There "
             "are no good, general methods for solving systems of more than "
             "one nonlinear equation.", "Numerical Recipes in C"]
]

def _format_quote(i, line_len=79, header=True):
    """Formats a quote string

    Parameters
    ----------
    i : int
       Index of the quote
    
    Returns
    -------
    quote : string
       Formatted quote
    """

    quote, author = _quotes[i]
    
    qtstr = str(i) + '.'
    start = len(qtstr)
    count = start
    for word in quote.split():
        count = count+1+len(word)
        if count > line_len:
            count = start+1 + len(word)
            word = '\n' + ' '*(start+1) + word
        else:
            word = ' ' + word
        qtstr += word

    qtstr += '\n'
    qtstr = qtstr.upper()

    # Add the author line
    qtstr += ' '*(start+2) +  ' - ' + author + '\n'

    # Add header/footer
    footer = '*'*line_len
    if header:
        header = footer + '\n'
    else:
        header = ''
    
    return header + qtstr + footer
    

def get_quote(i=None,get_all=False):
    """Get a quote string.

    Parameters
    ----------
    i : int, optional
        Index of the quote. If not provided a random one will be selected.
    get_all : bool, default=False
        If True all quotes will be returned, otherwise just one.
    
    Returns
    -------
    quote_string : string
        Quotes in string format, ready to be printed.
    """
    if get_all:
        quote = ''
        for i in range(1,len(_quote)):
            quote += _format_quote(i, header=(i==0))
        return quote

    if i is None:
        i = random.randint(1,len(_quotes)-1)

    return _format_quote(i)
    
    
def print_quote(i=None):
    """Print a quote message

    Parameters
    ----------
    i : int, optional
        Index of the quote. If not provided a random one will be selected.
    """
    quote = get_quote(i)
    print(quote)