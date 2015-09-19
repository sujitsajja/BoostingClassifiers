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
public class RealAdaBoosting
{
    public static void main(String[] args) throws FileNotFoundException
    {
        Scanner sc = new Scanner(new File("input.txt"));
        FileOutputStream fout = new FileOutputStream("outputRealAdaBoosting.txt");
        PrintStream ps = new PrintStream(fout);
        System.setOut(ps);
        int T = sc.nextInt();
        int n = sc.nextInt();
        double epsilon = sc.nextDouble();
        float[] x = new float[n];
        int[] y = new int[n];
        double[] p = new double[n];
        double[] preNormalizedP = new double[n];
        double[] classificationProbability = new double[n];
        double weights1,weights2,error,normalizationFactor=0;
        float threshold,bound=1;
        int[] weakclassifier = new int[2];
        double[] ft = new double[n];
        for(int i=0;i<n;i++)
            x[i]=sc.nextFloat();
        for(int i=0;i<n;i++)
            y[i]=sc.nextInt();
        for(int i=0;i<n;i++)
            p[i]=sc.nextDouble();
        for(int i=0;i<T;i++)
        {
            int errorCount=0;
            normalizationFactor = 0;
            System.out.println("\n\n");
            System.out.println("Iteration - "+(i+1));
            threshold = calculateThreshold(x,y,p);
            weakclassifier = calculateWeakClassifier(x,y,p,threshold);
            if(weakclassifier[0] == 1)
                System.out.println("Selected weak classifier : I(x>"+threshold+")");
            else if(weakclassifier[0] == 2)
                System.out.println("Selected weak classifier : I(x<"+threshold+")");
            classificationProbability = calculateClassficationProbability(x,y,p,threshold,weakclassifier[0]);
            error = Math.sqrt(classificationProbability[0]*classificationProbability[3])+Math.sqrt(classificationProbability[1]*classificationProbability[2]);
            System.out.println("Error : "+error);
            weights1=0.5*Math.log((classificationProbability[0]+epsilon)/(classificationProbability[3]+epsilon));
            weights2=0.5*Math.log((classificationProbability[2]+epsilon)/(classificationProbability[1]+epsilon));
            System.out.println("Weights : "+weights1+","+weights2);
            preNormalizedP = calculatePrenormalizedProbability(threshold,weakclassifier[0],p,x,y,weights1,weights2);
            for(int j=0;j<n;j++)
                normalizationFactor += preNormalizedP[j];
            System.out.println("Normalization factor : "+normalizationFactor);
            for(int j=0;j<n;j++)
                p[j] /= normalizationFactor;
            System.out.print("Probabilities after normalization : ");
            for(int j=0;j<n;j++)
                System.out.print(p[j]+" ");
            System.out.println();
            System.out.print("Values of ft(xi) : ");
            ft = calculateft(threshold,weakclassifier[0],weights1,weights2,ft,p,x,y);
            for(int j=0;j<n;j++)
            {
                if(j!=0)
                    System.out.print(" , ");
                System.out.print(ft[j]);
                if(ft[j]<0&&y[j]>0)
                    errorCount++;
                if(ft[j]>0&&y[j]<0)
                    errorCount++;
            }
            System.out.println();
            System.out.println("Error of boosted classifier : "+(float)errorCount/n);
            bound *= normalizationFactor;
            System.out.println("Bound : "+bound);
        }
    }
    public static float calculateThreshold(float[] x,int[] y,double[] p)
    {
        int n = x.length;
        ArrayList<Float> threshold = new ArrayList<Float>();
        ArrayList<Double> errorProbability = new ArrayList<Double>();
        double minimumError=100000.0;
        int minimumErrorIndex=0;
        for(int i=0;i<n-1;i++)
        {
            if(y[i]!=y[i+1])
                threshold.add((float)(x[i]+x[i+1])/2);
        }
        for(int i=0;i<threshold.size();i++)
        {
            float tempThreshold = threshold.get(i);
            double tempError1,tempError2;
            double[] classificationProbability = new double[4];
            int j=0;
            for(;x[j]<tempThreshold;j++)
            {
                if(y[j]>0)
                    classificationProbability[2] += p[j];
                else
                    classificationProbability[1] += p[j];
            }
            for(;j<n;j++)
            {
                if(y[j]<0)
                    classificationProbability[3] += p[j];
                else
                    classificationProbability[0] += p[j];
            }
            tempError1 = Math.sqrt(classificationProbability[0]*classificationProbability[3])+Math.sqrt(classificationProbability[1]*classificationProbability[2]);
            for(j=0;j<4;j++)
                classificationProbability[j] = 0;
            j =0;
            for(;x[j]<tempThreshold;j++)
            {
                if(y[j]<0)
                    classificationProbability[3] += p[j];
                else
                    classificationProbability[0] += p[j];
            }
            for(;j<n;j++)
            {
                if(y[j]>0)
                    classificationProbability[2] += p[j];
                else
                    classificationProbability[1] += p[j];
            }
            tempError2 = Math.sqrt(classificationProbability[0]*classificationProbability[3])+Math.sqrt(classificationProbability[1]*classificationProbability[2]);
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
    public static int[] calculateWeakClassifier(float[] x,int[] y,double[] p,float threshold)
    {
        int n = x.length;
        double tempError1,tempError2;
        int errorCount1=0,errorCount2=0;
        double[] classificationProbability = new double[4];
        int[] tempError = new int[2];
        int j=0;
        for(;x[j]<threshold;j++)
        {
            if(y[j]>0)
            {
                classificationProbability[2] += p[j];
                errorCount1++;
            }
            else
                classificationProbability[1] += p[j];
        }
        for(;j<n;j++)
        {
            if(y[j]<0)
            {
                classificationProbability[3] += p[j];
                errorCount1++;
            }
            else
                classificationProbability[0] += p[j];
        }
        tempError1 = Math.sqrt(classificationProbability[0]*classificationProbability[3])+Math.sqrt(classificationProbability[1]*classificationProbability[2]);
        for(j=0;j<4;j++)
            classificationProbability[j] = 0;
        j=0;
        for(;x[j]<threshold;j++)
        {
            if(y[j]<0)
            {
                classificationProbability[3] += p[j];
                errorCount2++;
            }
            else
                classificationProbability[0] += p[j];
        }
        for(;j<n;j++)
        {
            if(y[j]>0)
            {
                classificationProbability[2] += p[j];
                errorCount2++;
            }
            else
                classificationProbability[1] += p[j];
        }
        tempError2 = Math.sqrt(classificationProbability[0]*classificationProbability[3])+Math.sqrt(classificationProbability[1]*classificationProbability[2]);
        if(tempError1<tempError2)
        {
            tempError[0]=1;
            tempError[1]=errorCount1;
        }
        else
        {
            tempError[0]=2;
            tempError[1]=errorCount2;
        }
        return tempError;
    }
    public static double[] calculateClassficationProbability(float[] x,int[] y,double[] p,float threshold,int weakclassifier)
    {
        int n = x.length;
        double[] classificationProbability = new double[4];
        if(weakclassifier == 1)
        {
            int j=0;
            for(;x[j]<threshold;j++)
            {
                if(y[j]>0)
                    classificationProbability[2] += p[j];
                else
                    classificationProbability[1] += p[j];
            }
            for(;j<n;j++)
            {
                if(y[j]>0)
                    classificationProbability[0] += p[j];
                else
                    classificationProbability[3] += p[j];
            }
        }
        else if(weakclassifier == 2)
        {
            int j=0;
            for(;x[j]<threshold;j++)
            {
                if(y[j]>0)
                    classificationProbability[0] += p[j];
                else
                    classificationProbability[3] += p[j];
            }
            for(;j<n;j++)
            {
                if(y[j]>0)
                    classificationProbability[2] += p[j];
                else
                    classificationProbability[1] += p[j];
            }
        }
        return classificationProbability;
    }
    public static double[] calculatePrenormalizedProbability(float threshold,int weakclassifier,double[] p,float[] x,int[] y,double weight1,double weight2)
    {
        int n = x.length;
        if(weakclassifier == 1)
        {
            int j=0;
            for(;x[j]<threshold;j++)
            {
                if(y[j]>0)
                    p[j] = p[j]/Math.exp(weight2);
                else
                    p[j] = p[j]*Math.exp(weight2);
            }
            for(;j<n;j++)
            {
                if(y[j]>0)
                    p[j] = p[j]/Math.exp(weight1);
                else
                    p[j] = p[j]*Math.exp(weight1);
            }
        }
        else if(weakclassifier == 2)
        {
            int j=0;
            for(;x[j]<threshold;j++)
            {
                if(y[j]>0)
                    p[j] = p[j]/Math.exp(weight1);
                else
                    p[j] = p[j]*Math.exp(weight1);
            }
            for(;j<n;j++)
            {
                if(y[j]>0)
                    p[j] = p[j]/Math.exp(weight2);
                else
                    p[j] = p[j]*Math.exp(weight2);
            }
        }
        return p;
    }
    public static double[] calculateft(float threshold,int weakclassifier,double weight1,double weight2,double[] ft,double[] p,float[] x,int[] y)
    {
        int n = x.length;
        if(weakclassifier == 1)
        {
            int j=0;
            for(;x[j]<threshold;j++)
                ft[j] += weight2;
            for(;j<n;j++)
                ft[j] += weight1;
        }
        else if(weakclassifier == 2)
        {
            int j=0;
            for(;x[j]<threshold;j++)
                ft[j] += weight1;
            for(;j<n;j++)
                ft[j] += weight2;
        }
        return ft;
    }
}