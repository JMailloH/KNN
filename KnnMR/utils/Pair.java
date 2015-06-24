package org.apache.mahout.classifier.KnnMR.utils;

/**
 * Implements a simple pair.
 * @param F Type of the first element of the pair.
 * @param S Type of the second element of the pair.
 * @author diegoj based on Wikipedia Pair java class (http://en.wikipedia.org/wiki/Generics_in_Java)
 */
public class Pair <F, S> implements Comparable<Pair <F, S> >{

    /** First element of the pair. */
    protected F first;
    
    /** Second element of the pair. */
    protected S second;
   
    /**
     * Constructor of the pair
     * @param f First element.
     * @param s Second element.
     */
    public Pair(F f, S s)
    {
        first = f;
        second = s;
    }

    /**
     * Get first element of the pair.
     * @return First element of the pair.
     */
    public F first()
    {
        return first;
    }

    /**
     * Get second element of the pair.
     * @return Second element of the pair.
     */
    public S second()
    {
        return second;
    }

    /**
     * Get first element of the pair.
     * @return First element of the pair.
     */
    public F getFirst()
    {
        return first;
    }

    /**
     * Get second element of the pair.
     * @return Second element of the pair.
     */
    public S getSecond()
    {
        return second;
    }

    /**
     * Show par as String.
     * @return String representation of the pair.
     */
    @Override
    public String toString()
    {
        return "(" + first.toString() + ", " + second.toString() + ")";
    }

     /**
     * Set elements of the pair.
     * @param f New first element of the pair.
     * @param s New second element of the pair. 
     */
    public void set(F f, S s)
    {
        first = f;
        second = s;
    }

     /**
     * Invert the pair.
     * @return New pair whose first element is current second, and its second is current first.
     */    
    public Pair<S, F> invert()
    {
        return new Pair<S, F>(second, first);
    }
    
    public boolean equals(Pair<S, F> other)
    {
        return first.equals(other.first) && second.equals(other.second);
    }
    
    public boolean contains(Pair<S, F> other)
    {
        boolean straight = (first == other.first  &&  second == other.second);
        boolean inverted = (first == other.second  &&  second == other.first);
        return straight || inverted;
    }
   
    public boolean containsSomeElementOf(Pair<S, F> other)
    {
        boolean isFirst = first == other.first || first == other.second;
        boolean isSecond = second == other.first || second == other.second;
        return isFirst || isSecond;
    }
    
    @Override
    public int compareTo(Pair<F, S> o) {
    	Double a = (Double) this.getSecond();
    	Double b = (Double) o.getSecond();
    	return a.compareTo(b);
   }
 }
