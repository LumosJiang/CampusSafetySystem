package com.example.demo1.service;

import com.example.demo1.entity.Danger;
import com.example.demo1.entity.Video;
import com.example.demo1.mapper.VideoMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class VideoService {
    @Autowired
    private VideoMapper videoMapper;

    public void addVideo(int video_id,String path,String date){
        videoMapper.addVideo(video_id,path,date);
    }

    public List<Video> allVideo(){
        return videoMapper.allVideo();
    }

    public String getPath(int video_id){
        return videoMapper.getPath(video_id);
    }
    public String getPathByName(String name){
        return videoMapper.getPathByName(name);
    }

    public Video getDetail(int video_id){
        return videoMapper.getDetail(video_id);
    }

    public void delete(int video_id){
        videoMapper.delete(video_id);
    }

    public List<Danger> allDanger() {return videoMapper.allDanger();
    }
}
