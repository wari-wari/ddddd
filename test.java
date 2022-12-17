package chap3;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.*;

public class test{

    public static void main(String args[]){
    	Scanner sc = new Scanner(System.in);
    	
        List<String> strList = new ArrayList<String>();
       
        String target = sc.nextLine();
        
        while (sc.hasNext()) {
        	String s = sc.nextLine();
        	strList.add(s);
        }
        
        Collections.sort(strList);

        int dataLength = strList.size();
        System.out.println(binarySearch(dataLength, strList, target));
    }

    static int binarySearch(int dataLength, List<String> strList, String target){

        int begin = 0;
        int end = dataLength -1;
        int count = 0;

        while (begin <= end) {
        	count++;
            int mid = (begin + end) / 2;

            if(target.equals(strList.get(mid))) {
                return count;
            } else {
                
                if(target.compareTo(strList.get(mid)) < 0){
                    end = mid - 1;
                } 
                else {
                    begin = mid + 1;
                }
            }
        }
        return 1;
    }
}