package com.example.demo1.mapper;

import com.example.demo1.entity.Danger;
import com.example.demo1.entity.Video;
import org.apache.ibatis.annotations.Delete;
import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface VideoMapper {
    @Select("select * from video")
    List<Video> allVideo();

    @Insert("insert into video(video_id,path,date) values (#{video_id},#{path},#{date})")
    void addVideo(int video_id,String path,String date);

    @Select("select path from video where video_id=#{video_id}")
    String getPath(int video_id);

    @Select("select path from video where name=#{name}")
    String getPathByName(String name);

    @Select("select * from video where video_id=#{video_id}")
    Video getDetail(int video_id);

    @Delete("delete from video where video_id=#{video_id}")
    void delete(int video_id);

    @Select("select * from 异常事件列表")
    List<Danger> allDanger();
}
