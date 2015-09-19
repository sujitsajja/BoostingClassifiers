/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
/**
 *
 * @author sujit
 */

import java.io.*;
import java.util.*;
public class AdaBoosting
{
    public static void main(String[] args) throws FileNotFoundException
    {
        Scanner sc = new Scanner(new File("input.txt"));
        FileOutputStream fout = new FileOutputStream("outputBinaryAdaBoosting.txt");
        PrintStream ps = new PrintStream(fout);
        System.setOut(ps);
        int T = sc.nextInt();
        int n = sc.nextInt();
        double epsilon = sc.nextDouble();
        float[] x = new float[n];
        int[] y = new int[n];
        double[] p = new double[n];
        double[] preNormalizedP = new double[n];
        float threshold,bound=1;
        double weight,normalizationFactor=0;
        double error;
        double[] ft = new double[n];
        int weakclassifier;
        String boostedClassifier="";
        for(int i=0;i<n;i++)
            x[i]=sc.nextFloat();
        for(int i=0;i<n;i++)
            y[i]=sc.nextInt();
        for(int i=0;i<n;i++)
            p[i]=sc.nextDouble();
        for(int i=0;i<T;i++)
        {
            normalizationFactor = 0;
            System.out.println("\n\n");
            System.out.println("Iteration - "+(i+1));
            threshold = calculateThreshold(x,y,p);
            weakclassifier = calculateWeakClassifier(x,y,p,threshold);
            if(weakclassifier == 1)
                System.out.println("Selected weak classifier : I(x>"+threshold+")");
            else if(weakclassifier == 2)
                System.out.println("Selected weak classifier : I(x<"+threshold+")");
            error = calculateError(x,y,p,threshold,weakclassifier);
            System.out.println("Error : "+error);
            weight = 0.5*Math.log((1-error)/error);
            System.out.println("Weight : "+weight);
            preNormalizedP = calculatePrenormalizedProbability(threshold,weakclassifier,p,x,y,weight);
            for(int j=0;j<n;j++)
                normalizationFactor += preNormalizedP[j];
            System.out.println("Normalization factor : "+normalizationFactor);
            for(int j=0;j<n;j++)
                p[j] /= normalizationFactor;
            System.out.print("Probabilities after normalization : ");
            for(int j=0;j<n;j++)
                System.out.print(p[j]+" ");
            System.out.println();
            if(i!=0)
                boostedClassifier += " + ";
            if(weakclassifier == 1)
                boostedClassifier += weight+" * I(x>"+threshold+")";
            else if(weakclassifier == 2)
                boostedClassifier += weight+" * I(x<"+threshold+")";
            System.out.println("Boosted classifier : "+boostedClassifier);
            ft = calculateft(threshold,weakclassifier,weight,ft,x,y);
            int errorCount=0;
            for(int j=0;j<n;j++)
            {
                if(ft[j]<0 && y[j]>0)
                    errorCount++;
                else if(ft[j]>0 && y[j]<0)
                    errorCount++;
            }
            System.out.println("Error of boosted classifier : "+(double)errorCount/n);
            bound *= normalizationFactor;
            System.out.println("Bound : "+bound);
        }
    }
    public static float calculateThreshold(float[] x,int[] y,double[] p)
    {
        ArrayList<Float> threshold = new ArrayList<Float>();
        ArrayList<Double> errorProbability = new ArrayList<Double>();
        double minimumError=100000.0;
        int minimumErrorIndex=0;
        int n = x.length;
        for(int i=0;i<n-1;i++)
        {
            if(y[i]!=y[i+1])
                threshold.add((float)(x[i]+x[i+1])/2);
        }
        for(int i=0;i<threshold.size();i++)
        {
            float tempThreshold = threshold.get(i);
            double tempError1=0;
            double tempError2=0;
            int j=0;
            for(;x[j]<tempThreshold;j++)
            {
                if(y[j]>0)
                    tempError1 += p[j];
            }
            for(;j<n;j++)
            {
                if(y[j]<0)
                    tempError1 += p[j];
            }
            j=0;
            for(;x[j]<tempThreshold;j++)
            {
                if(y[j]<0)
                    tempError2 += p[j];
            }
            for(;j<n;j++)
            {
                if(y[j]>0)
                    tempError2 += p[j];
            }
            if(tempError1<tempError2)
                errorProbability.add(tempError1);
            else
                errorProbability.add(tempError2);
        }
        for(int i=0;i<errorProbability.size();i++)
        {
            if(errorProbability.get(i)<minimumError)
            {
                minimumError = errorProbability.get(i);
                minimumErrorIndex = i;
            }
        }
        return threshold.get(minimumErrorIndex);
    }
    public static double calculateError(float[] x,int[] y,double[] p,float threshold,int weakclassifier)
    {
        double tempError=0;
        int n = x.length;
        if(weakclassifier == 1)
        {
            int j=0;
            for(;x[j]<threshold;j++)
            {
                if(y[j]>0)
                    tempError += p[j];
            }
            for(;j<n;j++)
            {
                if(y[j]<0)
                    tempError += p[j];
            }
        }
        else if(weakclassifier ==2)
        {
            int j=0;
            for(;x[j]<threshold;j++)
            {
                if(y[j]<0)
                    tempError += p[j];
            }
            for(;j<n;j++)
            {
                if(y[j]>0)
                    tempError += p[j];
            }
        }
        return tempError;
    }
    public static int calculateWeakClassifier(float[] x,int[] y,double[] p,float threshold)
    {
        double tempError1=0;
        double tempError2=0;
        int j=0;
        int n = x.length;
        for(;x[j]<threshold;j++)
        {
            if(y[j]>0)
                tempError1 += p[j];
        }
        for(;j<n;j++)
        {
            if(y[j]<0)
                tempError1 += p[j];
        }
        j=0;
        for(;x[j]<threshold;j++)
        {
            if(y[j]<0)
                tempError2 += p[j];
        }
        for(;j<n;j++)
        {
            if(y[j]>0)
                tempError2 += p[j];
        }
        if(tempError1<tempError2)
            return 1;
        else
            return 2;
    }
    public static double[] calculatePrenormalizedProbability(float threshold,int weakclassifier,double[] p,float[] x,int[] y,double weight)
    {
        int n = x.length;
        if(weakclassifier == 1)
        {
            int j=0;
            for(;x[j]<threshold;j++)
            {
                if(y[j]>0)
                    p[j] = p[j]*Math.exp(weight);
                else
                    p[j] = p[j]/Math.exp(weight);
            }
            for(;j<n;j++)
            {
                if(y[j]<0)
                    p[j] = p[j]*Math.exp(weight);
                else
                    p[j] = p[j]/Math.exp(weight);
            }
        }
        else if(weakclassifier == 2)
        {
            int j=0;
            for(;x[j]<threshold;j++)
            {
                if(y[j]<0)
                    p[j] = p[j]*Math.exp(weight);
                else
                    p[j] = p[j]/Math.exp(weight);
            }
            for(;j<n;j++)
            {
                if(y[j]>0)
                    p[j] = p[j]*Math.exp(weight);
                else
                    p[j] = p[j]/Math.exp(weight);
            }
        }
        return p;
    }
    public static double[] calculateft(float threshold,int weakclassifier,double weight,double[] ft,float[] x,int[] y)
    {
        int n = x.length;
        if(weakclassifier == 1)
        {
            int j=0;
            for(;x[j]<threshold;j++)
                ft[j] -= weight;
            for(;j<n;j++)
                ft[j] += weight;
        }
        else if(weakclassifier == 2)
        {
            int j=0;
            for(;x[j]<threshold;j++)
                ft[j] += weight;
            for(;j<n;j++)
                ft[j] -= weight;
        }
        return ft;
    }
}